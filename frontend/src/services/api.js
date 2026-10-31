async function json(response){const data=await response.json();if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);return data}
export const predictStudent=(studentData)=>fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(studentData)}).then(json)
export const getGenAISummary=(context)=>fetch('/api/genai/summary',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({context})}).then(json)
export const sendChatMessage=(messages)=>fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages})}).then(json)
export const checkHealth=()=>fetch('/api/health').then(json)
