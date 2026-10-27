import {expect,test} from 'vitest';
import {latestStage,validationOutcome} from './evidence';
test('the evidence chain uses the latest matching job only for the selected experiment and stage',()=>{
 const jobs=[
  {status:'succeeded',created_at:1,metadata:{run_id:'mine',action:'train'}},
  {status:'failed',created_at:2,metadata:{run_id:'mine',action:'train'}},
  {status:'succeeded',created_at:3,metadata:{run_id:'other',action:'train'}},
  {status:'succeeded',created_at:4,metadata:{run_id:'mine',action:'export'}},
 ];
 expect(latestStage(jobs,'mine',['train'])?.status).toBe('failed');
 expect(latestStage(jobs,'mine',['validate'])).toBeNull();
 expect(latestStage(jobs,undefined,['train'])).toBeNull();
});
test('a completed process without JSON validation evidence is not promoted to correctness success',()=>{
 expect(validationOutcome('[stage 1/1]\n{"passed":true,"first_mismatch":null}')).toEqual({passed:true,mismatch:'null'});
 expect(validationOutcome('{"passed":false,"first_mismatch":2}')).toEqual({passed:false,mismatch:'2'});
 expect(validationOutcome('finished successfully').passed).toBeNull();
});
