"""End-to-end joint continuation with features from a real tiny Gemma decoder."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from draftfit.modeling.auto import AutoDraftModel
from draftfit.offline_capture.transformers import OfflineTransformersCapture
from draftfit.training.model_loading import convert_family_draft_config
from tests.test_dflash2_integration import tiny_config
from tests.test_family_joint_cpu import fixture
from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedConfig, Gemma4UnifiedTextConfig
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration


class HumanizerJointCPU(unittest.TestCase):
    def test_seeded_joint_campaign_resume_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fixture(root,seeded=True)
            config=Gemma4UnifiedTextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,
                num_hidden_layers=8,num_attention_heads=2,num_key_value_heads=1,
                head_dim=16,global_head_dim=16,num_global_key_value_heads=1,
                sliding_window=4,final_logit_softcapping=30.0)
            target=Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedConfig(text_config=config)).eval()
            target.save_pretrained(root/'target')
            teacher=OfflineTransformersCapture(target)
            teacher.set_capture_layers([1,5],capture_method='dspark')
            for split in ('train','holdout'):
                for file in sorted((root/split).glob('*.ckpt')):
                    ids=torch.randint(2,63,(1,8))
                    result=teacher.capture(input_ids=ids,attention_mask=torch.ones_like(ids),loss_mask=torch.ones_like(ids))
                    torch.save(dict(input_ids=ids[0],loss_mask=result.loss_mask[0],
                        hidden_states=result.hidden_states[0].to(torch.bfloat16),
                        target_last_hidden_states=result.last_hidden_states[0].to(torch.bfloat16)),file)
            seed_config=convert_family_draft_config(tiny_config(layer_types=['sliding_attention','full_attention'],
                sliding_window=4,final_logit_softcapping=30.0),'dflash')
            seed=AutoDraftModel.from_config(seed_config)
            seed.save_pretrained(root/'seed')
            seed_config.to_json_file(root/'draft.json')
            base=json.loads((root/'base.json').read_text())
            base['training'].update(num_epochs=3,max_steps=2,total_steps=2,eval_interval=1)
            (root/'base.json').write_text(json.dumps(base))
            env=os.environ.copy()
            env.update(SPECFORGE_DEVICE='cpu',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                PYTHONPATH=str(Path(__file__).resolve().parents[1]),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            for k in ('RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):env.pop(k,None)
            previous=None
            for step in (1,2):
                folder=root/f'milestone-{step}';folder.mkdir()
                receipt=folder/'result.json'
                env.update(DRAFTFIT_JOB_OUTPUT=str(folder),DRAFTFIT_JOB_RESULT=str(receipt),
                    DRAFTFIT_INPUT_RESULTS=json.dumps({'previous':str(previous)}) if previous else '{}')
                args=[sys.executable,'-m','draftfit.campaign_joint','--manifest',str(root/'family.json'),'--stop-at',str(step)]
                if previous:args+=['--resume-job','previous']
                result=subprocess.run(args,cwd=root,env=env,capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                payload=json.loads(receipt.read_text())
                self.assertEqual(payload['step'],step)
                self.assertEqual({a['strategy'] for a in payload['arms']},{'dspark','dflash','dflash2'})
                for arm in payload['arms']:
                    restored=AutoDraftModel.from_pretrained(arm['export'])
                    self.assertEqual(restored.config.layer_types,['sliding_attention','full_attention'])
                    self.assertEqual(restored.config.final_logit_softcapping,30.0)
                    self.assertFalse(torch.equal(restored.fc.weight.float(),seed.fc.weight.float()))
                previous=receipt
