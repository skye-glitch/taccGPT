import InferenceBar from "./InferenceBar"
import DragNDrop from "./DragNDrop"
import { useState, React } from "react"

function Ranking() {

  function updateData(newData) {
    setData(newData)
  }
  function updatePrompt(newPrompt) {
    setPrompt(newPrompt)
  }

  const [prompt, setPrompt] = useState('');
  const numAnswers = 4;
  const [data, setData] = useState([{"group":"group1", "items":[]},
                                    {"group":"group2", "items":[]},
                                    {"group":"group3", "items":[]},
                                    {"group":"group4", "items":[]},
                                    {"group":"Rank 1 (best)", "items":[]},
                                    {"group":"Rank 2", "items":[]},
                                    {"group":"Rank 3", "items":[]},
                                    {"group":"Rank 4 (worst)", "items":[]}]);

  return (
    <div className="App-header">
      <div className='APP-title'>
        Ranking outputs
      </div>
      <InferenceBar data={data} numAnswers={numAnswers} updateData={updateData} updatePrompt={updatePrompt} prompt={prompt} />
      <DragNDrop data={data} numAnswers={numAnswers} prompt={prompt} updateData={updateData} updatePrompt={updatePrompt} />
    </div>
        
  )
}

export default Ranking;