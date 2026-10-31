import { Routes, Route, Link } from 'react-router-dom'
import Input from './pages/Input'
import Results from './pages/Results'
import GenAI from './pages/GenAI'
import Metrics from './pages/Metrics'
export default function App(){return <><nav><Link to="/">Student Success Predictor</Link><span><Link to="/input">Input</Link><Link to="/metrics">Metrics</Link><Link to="/genai">GenAI</Link></span></nav><main><Routes><Route path="/" element={<Input/>}/><Route path="/input" element={<Input/>}/><Route path="/results" element={<Results/>}/><Route path="/genai" element={<GenAI/>}/><Route path="/metrics" element={<Metrics/>}/></Routes></main></>}
