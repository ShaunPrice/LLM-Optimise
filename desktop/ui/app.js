const invoke = window.__TAURI__.core.invoke;
const form = document.querySelector('#launch');
const status = document.querySelector('#status');
const button = document.querySelector('#start');
function message(value, error=false) { status.textContent=value; status.classList.toggle('error',error); }
invoke('defaults').then(value=>{document.querySelector('#python').value=value.interpreter||''; document.querySelector('#workspace').value=value.workspace||'';}).catch(error=>message(String(error),true));
form.addEventListener('submit',async event=>{event.preventDefault();button.disabled=true;message('Starting your local laboratory…');try{const value=await invoke('start_service',{interpreter:document.querySelector('#python').value,workspace:document.querySelector('#workspace').value});message(`Laboratory running at ${value.url}. Closing its window keeps it available in the tray.`);}catch(error){message(String(error),true);}finally{button.disabled=false;}});
document.querySelector('#stop').addEventListener('click',async()=>{try{await invoke('stop_service');message('Local service stopped and owned models released.');}catch(error){message(String(error),true);}});
