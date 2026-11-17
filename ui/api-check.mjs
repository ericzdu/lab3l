import {test} from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {mkdtempSync,writeFileSync,symlinkSync,unlinkSync,readFileSync,rmSync,existsSync} from 'node:fs';
import path from 'node:path';
import {tmpdir} from 'node:os';

const base=process.env.LAB3L_UI_URL||'http://localhost:5173';
const root=process.env.LAB3L_REPO||path.resolve('..');
const python=process.env.LAB3L_PYTHON||path.join(root,'shadow-hand-teleop/.venv',process.platform==='win32'?'Scripts/python.exe':'bin/python');
const reader=JSON.parse(execFileSync(python,['scripts/read_repository_data.py'],{env:{...process.env,LAB3L_REPO:root,PYTHONDONTWRITEBYTECODE:'1'},encoding:'utf8'}));
const file=p=>`${base}/api/repository/file?path=${encodeURIComponent(p)}`;

test('API reflects the existing repository loader without generating records',async()=>{
 const response=await fetch(`${base}/api/repository`);assert.equal(response.status,200);
 const body=await response.json();assert.equal(body.commit,reader.commit);assert.deepEqual(body.records,reader.records);
});
test('Data endpoint rejects writes',async()=>{
 const r=await fetch(`${base}/api/repository`,{method:'POST',body:'{}'});assert.equal(r.status,405);
});
test('Existing recordings download with identical bytes',async()=>{
 for(const record of reader.records){const r=await fetch(file(record.path));assert.equal(r.status,200);assert.deepEqual(Buffer.from(await r.arrayBuffer()),readFileSync(path.join(root,'shadow-hand-teleop/data',record.path)))}
});
test('Traversal and absolute external file paths are rejected',async()=>{
 for(const p of ['../requirements.txt','../../README.md','../../ui/package.json','/etc/passwd'])assert.equal((await fetch(file(p))).status,404);
});
test('A JSON symlink escaping the dataset directory is rejected',async t=>{
 const data=path.join(root,'shadow-hand-teleop/data');if(!existsSync(data)){t.skip('No local data directory; fresh clone');return}
 const temp=mkdtempSync(path.join(tmpdir(),'lab3l-ui-api-'));const secret=path.join(temp,'outside.json');const link=path.join(data,`ui-test-${path.basename(temp)}.json`);
 try{writeFileSync(secret,'{"outside":"test-only"}');symlinkSync(secret,link);assert.equal((await fetch(file(path.basename(link)))).status,404)}finally{if(existsSync(link))unlinkSync(link);rmSync(temp,{recursive:true,force:true})}
});
