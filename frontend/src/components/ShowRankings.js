import axios from "axios";
import React, { useEffect, useState } from "react";
import {motion} from "framer-motion"

function ShowRankings(){
  const [rankings, setRankings] = useState([]);

  useEffect(()=>{
    fetchData()
  }, [])

  const fetchData = async() =>{
    axios.get("http://localhost:9990/get_all_rankings").then(res => {
      console.log(res.data.rankings)
      setRankings(res.data.rankings)
    })
  }

  function downloadDataJson(data, name="data.json") {
    const octData = new Blob([JSON.stringify(data)], {type: "octet-stream"})
    const octDataHref = URL.createObjectURL(octData);
    const a = Object.assign(document.createElement("a"),{
      href: octDataHref,
      download:name,
      style: "display:none"
    })
    document.body.appendChild(a)
  
    a.click()
    URL.revokeObjectURL(octDataHref)
    a.remove()
  }

  return (
    <motion.div className="rankings-container"
    initial={{opacity: 0}}
    animate={{opacity: 1}}
    transition={{delay:0.8, duration:0.8}}>
      <div className="rankings-wrapper">
        <motion.button className="motion-button"
        whileHover={{scale:1.1}}
        onClick={()=>downloadDataJson(rankings, "rankings.json")}>
          Download rankings.json
        </motion.button>
        <div className="table-wrapper">
          
          <table className="fl-table">
            <thead> 
              <tr >    
                <th>Index</th>
                <th>Date</th>
                <th>User</th>
                <th>Prompt</th>
                <th>Rank 1</th>
                <th>Rank 2</th>
                <th>Rank 3</th>
                <th>Rank 4</th>
              </tr>
            </thead> 

            <tbody>
              {console.log(rankings)}
              {rankings.map((element, index) => (
                  <tr>
                    {console.log(index)}
                    {console.log(element.date)}
                    <td>{index}</td>
                    <td>{element.date}</td>
                    <td>{element.user}</td>
                    <td>{element.prompt}</td>
                    {element.answers.map((answer,answerIdx)=>(
                      <td>{JSON.stringify(answer)}</td>
                    ))}
                  </tr>
              ))}
            </tbody>
        </table>
        </div>
      </div>
    </motion.div>
  )
}
export default ShowRankings;