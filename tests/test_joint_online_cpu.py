"""Actual joint online CLI updates, frozen teacher, and exact continuation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from tests.test_family_joint_cpu import fixture


def online_fixture(root, seeded, gemma=False):
    manifest = fixture(root, seeded)
    target = Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32,
        intermediate_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, tie_word_embeddings=False))
    if gemma:
        from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig, Gemma4UnifiedTextConfig
        from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration
        config=Gemma4UnifiedTextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
            num_hidden_layers=8,num_attention_heads=2,num_key_value_heads=1,
            head_dim=16,global_head_dim=16,num_global_key_value_heads=1,
            sliding_window=4,final_logit_softcapping=30.0)
        target=Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(text_config=config))
    target.save_pretrained(root / 'target')
    for split, count, shift in [('train', 4, 0), ('eval', 2, 12)]:
        rows = [dict(input_ids=list(range(1+shift+i, 9+shift+i)),
                     loss_mask=[0,1,1,1,1,1,1,0]) for i in range(count)]
        (root / f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    base = json.loads((root/'base.json').read_text())
    base['model']['target_backend'] = 'transformers'
    base['training']['joint_online_capture'] = True
    base['training']['total_steps'] = 2
    base['data'] = dict(prompts_path=str(root/'train.jsonl'),
                        eval_data_path=str(root/'eval.jsonl'), max_length=8)
    (root/'base.json').write_text(json.dumps(base))
    return manifest


class JointOnlineCPU(unittest.TestCase):
    execution_device = "cpu"
    def cli(self, root, *args, success=True):
        env = os.environ.copy()
        env.update(SPECFORGE_DEVICE=self.execution_device, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        for k in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'): env.pop(k,None)
        p = subprocess.run([sys.executable,'-m','draftfit.cli',*args],cwd=root,
                           env=env,text=True,capture_output=True,timeout=120)
        if success: self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        else: self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)
        return p

    def exercise(self, seeded, gemma=False):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); manifest=online_fixture(root,seeded,gemma)
            original=(root/'target/model.safetensors').read_bytes()
            self.cli(root,'family','--manifest','family.json','--joint','--plan')
            self.assertFalse((root/'outputs').exists())
            self.cli(root,'family','--manifest','family.json',success=False)
            self.cli(root,'train','-c','base.json','--plan',success=False)
            self.cli(root,'family','--manifest','family.json','--joint')
            full=torch.load(root/'outputs/joint-step2.pt',weights_only=True)
            initial=torch.load(root/'outputs/joint-step0.pt',weights_only=True)
            for arm in full['drafts']:
                self.assertTrue(any(not torch.equal(v,initial['drafts'][arm][k])
                                    for k,v in full['drafts'][arm].items()))
            if gemma:
                from draftfit.export.to_hf import export_to_hf
                from draftfit.modeling.auto import AutoDraftModel
                for arm in full['drafts']:
                    checkpoint=root/'outputs'/arm/f'{arm}-latest'
                    destination=root/'exports'/arm
                    export_to_hf(str(checkpoint),str(checkpoint/'config.json'),str(destination),
                                 embedding_source=str(root/'target'))
                    restored=AutoDraftModel.from_pretrained(destination).state_dict()
                    for name,value in full['drafts'][arm].items():
                        self.assertTrue(torch.equal(value,restored[name].cpu()),(arm,name))
            curves=[json.loads(s) for s in (root/'outputs/learning-curves.jsonl').read_text().splitlines()]
            # 2 initial eval + 2 training microbatches; never multiplied by 3 arms.
            first=[r for r in curves if r['kind']=='capture' and r['step']==1]
            self.assertEqual({r['capture_calls'] for r in first},{4})
            self.assertEqual(original,(root/'target/model.safetensors').read_bytes())
            manifest['output_dir']='resumed';(root/'family.json').write_text(json.dumps(manifest))
            self.cli(root,'family','--manifest','family.json','--joint','--resume-from',str(root/'outputs/joint-step1.pt'))
            resumed=torch.load(root/'resumed/joint-step2.pt',weights_only=True)
            for arm in full['drafts']:
                for k,v in full['drafts'][arm].items():
                    self.assertTrue(torch.equal(v,resumed['drafts'][arm][k]),(arm,k))
            # A changed source cannot silently resume the previous run.
            rows=(root/'train.jsonl').read_text().splitlines(); row=json.loads(rows[0]); row['input_ids'][0]=33
            rows[0]=json.dumps(row);(root/'train.jsonl').write_text('\n'.join(rows)+'\n')
            manifest['output_dir']='changed';(root/'family.json').write_text(json.dumps(manifest))
            p=self.cli(root,'family','--manifest','family.json','--joint','--resume-from',str(root/'outputs/joint-step1.pt'),success=False)
            self.assertIn('checkpoint does not match',p.stderr+p.stdout)
            self.assertFalse(list((root/'outputs').rglob('capture_manifest.json')))
            self.assertFalse(list((root/'outputs').rglob('*.ckpt')))

    def test_scratch(self): self.exercise(False)
    def test_existing(self): self.exercise(True)

    def test_gemma_existing(self): self.exercise(True, gemma=True)

    def test_invalid_options(self):
        from draftfit.family import build_family_plan
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); online_fixture(root,False)
            base=json.loads((root/'base.json').read_text())
            for section,key,value in [('model','aux_hidden_state_layer_ids',[0,1,2]),
                                       ('training','compact_teacher',True)]:
                changed=json.loads(json.dumps(base)); changed[section][key]=value
                (root/'base.json').write_text(json.dumps(changed))
                with self.assertRaises(ValueError): build_family_plan(str(root/'family.json'))
