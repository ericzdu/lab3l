import {chromium} from '@playwright/test';
import {execFileSync} from 'node:child_process';
import {readFileSync,mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
const root=process.env.LAB3L_REPO||path.resolve('..');
const actual=JSON.parse(execFileSync(process.env.LAB3L_PYTHON||path.join(root,'shadow-hand-teleop/.venv/bin/python'),['scripts/read_repository_data.py'],{env:{...process.env,LAB3L_REPO:root},encoding:'utf8'}));
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH,headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1100}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
try{
 await page.goto(process.env.LAB3L_UI_URL||'http://localhost:5173');
 await page.evaluate(()=>localStorage.setItem('lab3l-episodes-v1',JSON.stringify([{id:'demo_0004',frames:99999}])));
 await page.reload();await page.getByRole('heading',{name:'Teleoperation',exact:true}).waitFor();
 await page.waitForFunction(()=>document.body.innerText.includes('Local checkout:')&&!document.body.innerText.includes('Local checkout: not loaded'));
 assert.equal(await page.evaluate(()=>localStorage.getItem('lab3l-episodes-v1')),null);
 assert.equal(await page.getByText('demo_0004',{exact:true}).count(),0);
 assert.equal(await page.getByRole('button',{name:'Live recording unavailable'}).isDisabled(),true);
 assert.equal(await page.locator('canvas').count(),0);
 await page.screenshot({path:'preview-desktop.png',fullPage:true});
 await page.getByRole('button',{name:/^Datasets/}).click();
 assert.equal(await page.locator('tbody tr').count(),actual.records.length);
 if(actual.records.length){
  const expected=actual.records[0];await page.getByRole('button',{name:expected.path,exact:true}).click();
  const dialog=page.getByRole('dialog');await dialog.getByRole('heading',{name:expected.name,exact:true}).waitFor();
  if(expected.metadata?.task_outcome)await dialog.getByText(expected.metadata.task_outcome,{exact:true}).waitFor();
  if(expected.preview.length)assert.deepEqual(JSON.parse(await dialog.locator('pre').innerText()),expected.preview[0]);
  const temp=mkdtempSync(path.join(tmpdir(),'lab3l-download-'));try{
   const promise=page.waitForEvent('download');await dialog.getByRole('link',{name:'Download original file'}).click();const dl=await promise;const dest=path.join(temp,'download');await dl.saveAs(dest);assert.deepEqual(readFileSync(dest),readFileSync(path.join(root,'shadow-hand-teleop/data',expected.path)));
  }finally{rmSync(temp,{recursive:true,force:true})}
  await dialog.getByRole('button',{name:'Close',exact:true}).click();
 }
 await page.getByLabel('Search datasets').fill('no-such-dataset');await page.getByRole('heading',{name:actual.records.length?'No matching files':'No dataset files found'}).waitFor();await page.getByRole('button',{name:'Clear search'}).click();
 await page.screenshot({path:'preview-data.png',fullPage:true});
 await page.getByRole('button',{name:'Workflows',exact:true}).click();assert.equal(await page.getByText('Apple to plate',{exact:true}).count(),0);await page.getByText('Task labels & evaluation',{exact:true}).waitFor();
 await page.setViewportSize({width:390,height:844});
 for(const name of ['Teleoperation','Datasets','Workflows','Setup']){await page.getByRole('button',{name:new RegExp('^'+name)}).click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false,`Overflow on ${name}`)}
 await page.getByRole('button',{name:'Teleoperation',exact:true}).click();await page.screenshot({path:'preview-mobile.png',fullPage:true});
 await page.route('**/api/repository',route=>route.fulfill({status:503,contentType:'application/json',body:JSON.stringify({error:'Reader unavailable during verification'})}));
 await page.getByRole('button',{name:'Refresh local files',exact:true}).click();await page.getByRole('alert').waitFor();await page.getByRole('button',{name:/^Datasets/}).click();assert.equal(await page.locator('tbody tr').count(),0);await page.getByRole('heading',{name:'Repository data unavailable'}).waitFor();
 assert.deepEqual(errors,[]);console.log(JSON.stringify({passed:['real file listing','obsolete demo cache removal','disabled unsupported recording','no fabricated 3D scene','original metadata','actual numeric rows','byte-identical download','search','mobile layout on all pages','honest unavailable state'],records:actual.records.length,pageErrors:errors}));
}finally{await browser.close()}
