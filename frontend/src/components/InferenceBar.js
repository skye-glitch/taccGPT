import React, {useEffect, useState} from "react";
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css'; 

const InferenceBar = props => {
  const [prompt, setPrompt] = useState('');

  useEffect(() => {
    const textarea = document.querySelector('textarea')
    textarea.value = props.prompt===undefined?'':props.prompt
    setPrompt(props.prompt===undefined?'':props.prompt)
  }, [props.prompt])

  const getRandomQuestion = (e) => {
    e.preventDefault();

    var textarea = document.querySelector('textarea')

    const pendingClassName = 'loading-btn--pending';
    const successClassName = 'loading-btn--success';
    const failClassName    = 'loading-btn--fail';
    const elem = document.getElementsByClassName("loading-btn-wrapper random-question")[0].querySelector("button");
    elem.classList.add(pendingClassName);

    axios.get('/backend/get_random_question').then(res => {
      window.setTimeout(() => {
        elem.classList.remove(pendingClassName);
        const classNameToBeAdded = res.data.success?successClassName:failClassName;
        elem.classList.add(classNameToBeAdded);
      
        window.setTimeout(() => {
          elem.classList.remove(classNameToBeAdded)
          
          textarea.value = res.data.message
          setPrompt(res.data.message)
          
        }, 500);
    }, 500);})

  }

  const submitPrompt = (e) => {
    e.preventDefault();
    var form = document.querySelector('form');
    if(!form.checkValidity()) {
      var tmpSubmit = document.createElement('button')
      form.appendChild(tmpSubmit)
      tmpSubmit.click()
      form.removeChild(tmpSubmit)
    } else {
      const stateDuration = 1000;
      const pendingClassName = 'loading-btn--pending';
      const successClassName = 'loading-btn--success';
      const failClassName    = 'loading-btn--fail';
      const elem = document.getElementsByClassName("loading-btn-wrapper submit-prompt")[0].querySelector("button");
      elem.classList.add(pendingClassName);

      axios.post('/TACC_GPT/submit_prompt/',{'prompt':prompt,'numAnswers':props.numAnswers},{timeout: 30000000000}).then(res => {

      window.setTimeout(() => {
        elem.classList.remove(pendingClassName);
        const classNameToBeAdded = res.data.answers.length === props.numAnswers?successClassName:failClassName;
        elem.classList.add(classNameToBeAdded);
      
        window.setTimeout(() => {
          elem.classList.remove(classNameToBeAdded)
          console.assert(res.data.answers.length === props.numAnswers)
          let newData = JSON.parse(JSON.stringify(props.data));
          for(let i = 0; i < props.numAnswers*2; i++) {
            if(i < props.numAnswers/2) {
              newData[i] = {"group":"group"+i, "items":[]};
            } else if(i === props.numAnswers) {
              newData[i] = {"group":"Rank 1 (best)", "items":[]};
            } else if(i === props.numAnswers*2-1) {
              newData[i] = {"group":"Rank "+ String(props.numAnswers) +" (worst)", "items":[]};
            } else {
              newData[i] = {"group":"Rank "+String(i-props.numAnswers+1), "items":[]};
            }
          }
          for(let i = 0; i < props.numAnswers; i++) newData[i].items.splice(0,0,res.data.answers[i]);
          props.updateData(newData)
          props.updatePrompt(prompt)
        }, stateDuration);
      }, stateDuration);})
    }
  }

  // const handleButtonClicks = (e) => {
  //   e.preventDefault();

  //   if(formState.button === "submitPrompt") {
     
  //   } else if (formState.button === "getRandomQuestion") {
  //     console.log("Clicked random question button")
  //   }

  // }

  return (
    <div className="raning-wrapper">
      <div className="inference-wrapper" id="inference-wrapper">
        <form id="submit-prompt">
          <textarea type="text"
           placeholder="Please enter here"
           onChange={(e) => setPrompt(e.target.value)} 
           required/>
          <div className="loading-btn-wrapper random-question">
            <button className="loading-btn js_success-animation-trigger" onClick={getRandomQuestion}>
              <span className="loading-btn__text">
                Random question
              </span>
            </button>
          </div>
          <div className="loading-btn-wrapper submit-prompt">
            <button className="loading-btn js_success-animation-trigger" onClick={submitPrompt}>
              <span className="loading-btn__text">
                Run
              </span>
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default InferenceBar;