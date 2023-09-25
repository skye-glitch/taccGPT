import React, { useEffect, useState } from "react";
import "./table_styles.css";
import axios from "axios";
import {motion} from "framer-motion";



function ShowQAParis(){
  const [qaPairs, setQaPairs] = useState([]);

  useEffect(()=>{
    fetchData()
  }, [])

  const fetchData = async() =>{
    // http://localhost:9990/get_all_qa_pairs
    axios.get("/backend/get_all_qa_pairs").then(res => {
      // console.log(res.data.qaPairs)
      setQaPairs(res.data.qaPairs)
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
    <motion.div className="QA-pairs-container"
    initial={{opacity: 0}}
    animate={{opacity: 1}}
    transition={{delay:0.8, duration:0.8}}>
    <div className="QA-pairs-wrapper">
      <motion.button className="motion-button"
      whileHover={{scale:1.1}}
      onClick={()=>downloadDataJson(qaPairs, "qaPairs.json")}>
        Download qa_pairs.json
      </motion.button>
      <div className="table-wrapper">
        <table className="fl-table">
          <thead> 
            <tr >    
              <th>Index</th>
              <th>Date</th>
              <th>User</th>
              <th>Prompt</th>
              <th>Answer</th>
            </tr>
          </thead> 

          <tbody>
            {console.log(qaPairs)}
            {qaPairs.map((element, index) => (
                <tr key={index}>
                  <td>{index}</td>
                  <td>{element.date}</td>
                  <td>{element.user}</td>
                  <td>{element.prompt}</td>
                  <td>{element.answer}</td>
                </tr>
            ))}
          </tbody>
      </table>
      </div>
    </div>
    </motion.div>
  )
}

export default ShowQAParis;