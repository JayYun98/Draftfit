"""Actual Metal execution; deliberately separate from portable CPU suites."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
except ImportError:
    mx = None


def tiny_teacher():
    mx.random.seed(29)
    args = TextModelArgs(model_type='qwen3_5_text', hidden_size=32, intermediate_size=64,
                         vocab_size=64, num_hidden_layers=8, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=8, full_attention_interval=4,
                         linear_num_key_heads=2, linear_num_value_heads=4,
                         linear_key_head_dim=32, linear_value_head_dim=32,
                         rope_parameters={'rope_type':'default', 'rope_theta':10000,
                                          'mrope_section':[1,1,0], 'partial_rotary_factor':0.5})
    return TextModel(args).eval()


@unittest.skipIf(mx is None, 'MLX requires Apple Silicon runtime')
class MLXCaptureTest(unittest.TestCase):
    def test_capture_native_readout_final_norm_padding_and_state_isolation(self):
        from draftfit.offline_capture.mlx import OfflineMLXCapture
        model = tiny_teacher()
        teacher = OfflineMLXCapture(model)
        teacher.set_capture_layers([1, 7])
        ids = torch.tensor([[2,3,4,5,6], [0,0,7,8,9]])
        attn = torch.tensor([[1,1,1,1,1], [0,0,1,1,1]])
        batch = teacher.capture(input_ids=ids, attention_mask=attn, loss_mask=attn)
        for i, start in [(0,0), (1,2)]:
            x = mx.array(ids[i,start:].numpy()[None])
            native = model(x, cache=model.make_cache())
            last = mx.array(batch.last_hidden_states[i:i+1,start:].numpy())
            got = model.lm_head(last)
            np.testing.assert_allclose(np.asarray(got), np.asarray(native), rtol=1e-5, atol=2e-5)
            prenorm = mx.array(batch.hidden_states[i:i+1,start:,32:].numpy())
            np.testing.assert_allclose(np.asarray(model.model.norm(prenorm)), np.asarray(last), rtol=1e-5, atol=2e-5)
        self.assertTrue(torch.equal(batch.hidden_states[1,:2], torch.zeros(2,64)))
        again = teacher.capture(input_ids=ids, attention_mask=attn, loss_mask=attn)
        torch.testing.assert_close(again.hidden_states, batch.hidden_states, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            teacher.capture(input_ids=ids, attention_mask=torch.tensor([[1,0,1,1,1],[0,0,1,1,1]]), loss_mask=attn)
        with self.assertRaises(ValueError):
            teacher.capture(input_ids=ids, attention_mask=attn, loss_mask=torch.ones_like(ids))
        with self.assertRaises(ValueError):
            teacher.set_capture_layers([7,1])

    def test_hadamard_packed_embedding_and_head_basis(self):
        from draftfit.offline_capture.mlx_hadamard import Packed, fwht, install_packed
        mx.random.seed(7)
        w = mx.random.normal((8,512)).astype(mx.float16)
        signs = mx.where(mx.arange(512) % 3, 1.0, -1.0)
        arrays = mx.quantize(w, group_size=128, bits=2)
        linear = Packed(arrays, 512, signs)
        embedding = Packed(arrays, 512, signs, embedding=True)
        dense = linear.dense_rows(0,8)
        ids = mx.array([[1,3,7]])
        np.testing.assert_array_equal(np.asarray(embedding(ids)), np.asarray(dense[ids]))
        x = mx.random.normal((1,3,512)).astype(mx.float16)
        # A naive dequantized rotated head is intentionally far from the native readout.
        expected = linear(x)
        reconstructed = x @ dense.T
        np.testing.assert_allclose(np.asarray(reconstructed), np.asarray(expected), rtol=.06, atol=.1)
        np.testing.assert_allclose(np.asarray(fwht(fwht(x,512,signs),512,signs,inverse=True)), np.asarray(x), atol=.003, rtol=.003)
        from draftfit.modeling.target.quantized_parts import HadamardEmbedding, HadamardHead
        tensors = [torch.from_numpy(np.array(a)) for a in arrays]
        torch_signs = torch.from_numpy(np.array(signs))
        frozen_embedding = HadamardEmbedding(tensors, block=512, signs=torch_signs)
        frozen_head = HadamardHead(tensors, block=512, signs=torch_signs)
        np.testing.assert_allclose(frozen_embedding(torch.tensor([[1,3,7]])).numpy(), np.asarray(embedding(ids)), rtol=.003, atol=.003)
        inputs = torch.from_numpy(np.array(x)).requires_grad_(True)
        logits = frozen_head(inputs)
        np.testing.assert_allclose(logits.detach().numpy(), np.asarray(expected), rtol=.02, atol=.03)
        self.assertTrue(torch.equal(logits.argmax(-1), torch.from_numpy(np.array(expected)).argmax(-1)))
        logits.float().square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        before = frozen_head.weight.clone()
        frozen_head.to(dtype=torch.bfloat16)
        self.assertEqual(frozen_head.weight.dtype, torch.float16)
        torch.testing.assert_close(before, frozen_head.weight, rtol=0, atol=0)
        module = nn.Sequential(nn.Linear(512,8,bias=False))
        weights = {f'layers.0.{k}':v for k,v in zip(('weight','scales','biases'),arrays)}
        weights['layers.0.signs'] = signs
        record = {'path':'layers.0','block':512,'embedding':False,'dtype':'float16'}
        install_packed(module,[record],weights)
        module.load_weights(list(weights.items()),strict=True)
        weights['layers.0.signs'] = mx.zeros(512)
        with self.assertRaises(ValueError):
            install_packed(nn.Sequential(nn.Linear(512,8,bias=False)),[record],weights)

    def test_mlx_features_to_three_family_updates_and_export(self):
        from draftfit.offline_capture.mlx import OfflineMLXCapture
        from tests.test_family_joint_cpu import fixture, FamilyJointCPU
        from safetensors.torch import save_file
        model = tiny_teacher()
        teacher = OfflineMLXCapture(model)
        teacher.set_capture_layers([1,5])
        runner = FamilyJointCPU()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, True)
            save_file({'model.embed_tokens.weight':torch.from_numpy(np.array(model.model.embed_tokens.weight)),
                       'lm_head.weight':torch.from_numpy(np.array(model.lm_head.weight))},
                      str(root/'target/model.safetensors'))
            for split, count in [('train',4),('holdout',2)]:
                for i in range(count):
                    ids = torch.arange(1,9)[None] + i + (8 if split=='holdout' else 0)
                    mask = torch.ones_like(ids)
                    batch = teacher.capture(input_ids=ids, attention_mask=mask, loss_mask=mask)
                    torch.save({'input_ids':ids[0], 'loss_mask':mask[0],
                                'hidden_states':batch.hidden_states[0].to(torch.bfloat16),
                                'target_last_hidden_states':batch.last_hidden_states[0].to(torch.bfloat16)},
                               root/split/f'{i:02d}.ckpt')
            runner.cli(root,'family','--manifest','family.json','--joint')
            result = json.loads((root/'outputs/results.json').read_text())
            self.assertEqual([x['status'] for x in result], ['complete']*3)
            for arm in ['dspark','ce','selector']:
                stage = root/'outputs'/arm
                runner.cli(root,'export','--to','hf','--checkpoint',str(stage/f'{arm}-latest'),
                           '--draft-config',str(stage/'draft.json'),'--output-dir',str(stage/'hf'),
                           '--embedding-source',str(root/'target'))
                from draftfit.modeling.auto import AutoDraftModel
                reloaded = AutoDraftModel.from_pretrained(stage/'hf')
                self.assertTrue(all(torch.isfinite(x).all() for x in reloaded.parameters()))

    def test_strict_bonsai_pack_native_capture_and_reject_missing_weight(self):
        import dataclasses
        from mlx.utils import tree_flatten
        from mlx_vlm.models.qwen3_5.config import TextConfig
        from mlx_vlm.models.qwen3_5.language import LanguageModel
        from draftfit.offline_capture.mlx_hadamard import Packed, load_bonsai_text
        from draftfit.offline_capture.mlx import OfflineMLXCapture
        config = dataclasses.asdict(tiny_teacher().args)
        config.update(hidden_size=512, intermediate_size=1024, head_dim=128,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      rope_parameters={'rope_type':'default','rope_theta':10000,
                                       'mrope_section':[4,4,8],'partial_rotary_factor':.25})
        model = LanguageModel(TextConfig.from_dict(copy.deepcopy(config))).eval()
        records = []
        for name, module in list(model.named_modules()):
            if not isinstance(module,(nn.Linear,nn.Embedding)):
                continue
            width = module.weight.shape[-1]
            arrays = mx.quantize(module.weight.astype(mx.float16),group_size=128,bits=2)
            block = 512 if width % 512 == 0 else 0
            signs = mx.where(mx.arange(width)%3,1.0,-1.0) if block else None
            parts = name.split('.')
            parent = model
            for part in parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent,part)
            setattr(parent,parts[-1],Packed(arrays,block,signs,isinstance(module,nn.Embedding)))
            records.append({'path':name,'embedding':isinstance(module,nn.Embedding),
                            'block':block,'dtype':'float16'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            payload = {'schema_version':2,'model_type':'prism_hadamard_qwen35',
                       'text_config':config,'components':{'text':True,'vision':True,'mtp':False},
                       'base_model_type':'qwen3_5','tensor_namespace':'mlx-vlm-qwen3_5',
                       'gdn_activation_layout':'grouped',
                       'quantization':{'bits':2,'group_size':128,'mode':'affine'},'modules':records}
            (path/'config.json').write_text(json.dumps(payload))
            weights = {'language_model.'+k:v for k,v in tree_flatten(model.parameters())}
            weights['vision_tower.dummy.weight'] = mx.zeros(1)
            mx.save_safetensors(str(path/'model.safetensors'),weights)
            reloaded = load_bonsai_text(path)
            from draftfit.modeling.target.quantized_parts import load_target_parts
            from draftfit.export.to_hf import _load_embedding_tensor
            wrong_layout = dict(payload, gdn_activation_layout='ungrouped')
            (path/'config.json').write_text(json.dumps(wrong_layout))
            with self.assertRaisesRegex(ValueError, 'tensor/layout contract'):
                load_bonsai_text(path)
            with self.assertRaisesRegex(ValueError, 'tensor/layout contract'):
                load_target_parts(path)
            (path/'config.json').write_text(json.dumps(payload))
            frozen = load_target_parts(path,device='cpu',dtype=torch.bfloat16)
            embedding = _load_embedding_tensor(str(path),'model.embed_tokens.weight')
            self.assertEqual(embedding.shape,(64,512))
            np.testing.assert_allclose(embedding[[1,2,3,4]].numpy(),np.asarray(model.model.embed_tokens(mx.array([1,2,3,4]))),rtol=.003,atol=.003)
            teacher = OfflineMLXCapture.from_pretrained(path)
            teacher.set_capture_layers([1,3])
            ids = torch.tensor([[1,2,3,4]])
            batch = teacher.capture(input_ids=ids,attention_mask=torch.ones_like(ids),loss_mask=torch.ones_like(ids))
            x = mx.array(ids.numpy())
            native_last = model.model(x,cache=model.make_cache())
            native = model.lm_head(native_last)
            last = mx.array(batch.last_hidden_states.numpy()).astype(native_last.dtype)
            actual = reloaded.lm_head(last)
            np.testing.assert_array_equal(np.asarray(actual),np.asarray(native))
            # Exercise the actual provider branch, packed frozen readout, and
            # three-family trainer together; this is a tiny pack, not Bonsai 27B.
            from tests.test_family_joint_cpu import fixture, FamilyJointCPU
            from tests.test_dflash2_integration import tiny_config
            from draftfit.training.model_loading import convert_family_draft_config
            from draftfit.modeling.auto import AutoDraftModel
            from draftfit.inference.capture_manifest import build_capture_manifest, write_capture_manifest
            job = path/'joint';job.mkdir()
            fixture(job,True)
            draft = convert_family_draft_config(tiny_config(
                hidden_size=512,intermediate_size=1024,num_hidden_layers=1,head_dim=128,
                target_num_hidden_layers=4,target_layer_ids=[1,3],conv_group_size=16),'dspark')
            draft.to_json_file(str(job/'draft.json'))
            AutoDraftModel.from_config(draft).save_pretrained(job/'packed-seed')
            base=json.loads((job/'base.json').read_text())
            base['model'].update(target_model_path=str(path),draft_checkpoint_path=str(job/'packed-seed'))
            (job/'base.json').write_text(json.dumps(base))
            manifest=build_capture_manifest(strategy='dspark',capture_method='dspark',capture_layers=[1,3],
                feature_names=('input_ids','loss_mask','hidden_states','target_last_hidden_states'),
                target_model=str(path),target_revision=base['model']['target_revision'],
                target_feature='target_last_hidden_states',target_repr='hidden_state')
            for split,count in [('train',4),('holdout',2)]:
                write_capture_manifest(str(job/split/'capture_manifest.json'),manifest)
                for i in range(count):
                    tokens=torch.arange(1,9)[None]+i+(8 if split=='holdout' else 0)
                    mask=torch.ones_like(tokens)
                    features=teacher.capture(input_ids=tokens,attention_mask=mask,loss_mask=mask)
                    torch.save({'input_ids':tokens[0],'loss_mask':mask[0],
                                'hidden_states':features.hidden_states[0].to(torch.bfloat16),
                                'target_last_hidden_states':features.last_hidden_states[0].to(torch.bfloat16)},
                               job/split/f'{i:02d}.ckpt')
            runner=FamilyJointCPU()
            runner.cli(job,'family','--manifest','family.json','--joint')
            results=json.loads((job/'outputs/results.json').read_text())
            self.assertEqual([r['status'] for r in results],['complete']*3)
            for arm in ['dspark','ce','selector']:
                stage=job/'outputs'/arm
                runner.cli(job,'export','--to','hf','--checkpoint',str(stage/f'{arm}-latest'),
                    '--draft-config',str(stage/'draft.json'),'--embedding-source',str(path),
                    '--output-dir',str(stage/'packed-export'))
                restored=AutoDraftModel.from_pretrained(stage/'packed-export')
                self.assertTrue(all(torch.isfinite(p).all() for p in restored.parameters()))
            del weights['language_model.model.norm.weight']
            mx.save_safetensors(str(path/'model.safetensors'),weights)
            with self.assertRaises(ValueError):
                load_bonsai_text(path)

    def test_prepare_cli_persists_contract_and_rejects_tokenizer_drift(self):
        import dataclasses
        import subprocess
        import sys
        from mlx.utils import tree_flatten
        from draftfit.offline_capture.prepare_mlx import sha256
        from draftfit.inference.capture_manifest import load_capture_manifest
        model = tiny_teacher()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            target=root/'target';target.mkdir()
            (target/'config.json').write_text(json.dumps(dataclasses.asdict(model.args)))
            (target/'tokenizer.json').write_text('{}')
            mx.save_safetensors(str(target/'model.safetensors'),dict(tree_flatten(model.parameters())))
            draft={'hidden_size':32,'vocab_size':64,'dflash_config':{'target_layer_ids':[1,7]}}
            (root/'draft.json').write_text(json.dumps(draft))
            header={'kind':'provenance','target_revision':'a'*40,'tokenizer_sha256':sha256(target/'tokenizer.json')}
            row={'id':'train:0','input_ids':[1,2,3,4],'loss_mask':[0,1,1,1]}
            data=root/'data.jsonl';data.write_text(json.dumps(header)+'\n'+json.dumps(row)+'\n')
            command=[sys.executable,'-m','draftfit.offline_capture.prepare_mlx','--target',str(target),
                     '--target-revision','a'*40,'--draft-config',str(root/'draft.json'),
                     '--data',str(data),'--output',str(root/'features')]
            result=subprocess.run(command,text=True,capture_output=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            contract=load_capture_manifest(str(root/'features/capture_manifest.json'))
            self.assertEqual(contract['producer']['backend'],'mlx')
            sample=torch.load(root/'features/000000.ckpt',weights_only=True)
            self.assertEqual(sample['hidden_states'].shape,(4,64))
            self.assertEqual(sample['target_last_hidden_states'].shape,(4,32))
            self.assertTrue((root/'features/capture_complete.json').is_file())
            result=subprocess.run(command,text=True,capture_output=True,timeout=60)
            self.assertNotEqual(result.returncode,0)
            (target/'tokenizer.json').write_text('{"changed":true}')
            command[-1]=str(root/'drift')
            result=subprocess.run(command,text=True,capture_output=True,timeout=60)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/'drift').exists())
