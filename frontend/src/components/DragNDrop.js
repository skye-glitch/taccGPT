import React, {useState, useRef, useEffect} from "react";
import axios from 'axios';

function DragNDrop(props){
  const dragItem = useRef();
  const dragNode = useRef();
  const [dragging, setDragging] = useState(false);
  const [show, setShow] = useState(false);
  const [user, setUser] = useState(null);

  function constructDefaultPropsData() {
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
    return newData
  }

  // Animation experiment starts here
  // function handleButtonAnimation(ev) {
  //   const stateDuration = 1500;
  //   const pendingClassName = 'loading-btn--pending';
  //   const successClassName = 'loading-btn--success';
  //   const failClassName    = 'loading-btn--fail';
  //   const elem = ev.target;
  //   elem.classList.add(pendingClassName);
    
  //   window.setTimeout(() => {
  //       elem.classList.remove(pendingClassName);
  //       elem.classList.add(successClassName);
      
  //       window.setTimeout(() => elem.classList.remove(successClassName), stateDuration);
  //   }, stateDuration);
  // }
  // Animation experiment ends here


  useEffect(()=>{
    const isAllRanked=(dataList)=>{
      let count = 0;
      for(let i = 0; i < props.data.length/2; i++){
        if(i < props.data.length/2){
          if(dataList[i].items.length !== 0) return false;
        } else {
          count+=dataList[i].items.length;
        }
      }
      return count === props.data.length/2;
    }
    setShow(isAllRanked(props.data));
  }, [props.data])

  const onDragStartHandle = (e, params) => {
    dragItem.current = params;
    dragNode.current = e.target;
    dragNode.current.addEventListener('dragend', dragEndHandle)
    setTimeout(()=>{
      setDragging(true)
    }, 0)
  }

  const dragEndHandle = () => {
    dragNode.current.removeEventListener('dragend', dragEndHandle);
    setDragging(false);
    dragItem.current = null;
    dragNode.current = null;
  }

  const onDragEnterHandle = (e, params) => {
    const currentItem = dragItem.current;
    if(e.target !== currentItem){
      // setList(oldList => {
      //   let newList = JSON.parse(JSON.stringify(oldList));
      //   newList[params.grpIdx].items.splice(params.itemIdx, 0, newList[currentItem.grpIdx].items.splice(currentItem.itemIdx,1)[0])
      //   dragItem.current = params;
      //   setTimeout(()=>{
      //     setShow(isAllRanked(newList));
      //   }, 1000)
      //   props.updateData(newList)
      //   return newList
      // })
      let newList = JSON.parse(JSON.stringify(props.data));

      newList[params.grpIdx].items.splice(params.itemIdx, 0, newList[currentItem.grpIdx].items.splice(currentItem.itemIdx,1)[0])
      props.updateData(newList)
      dragItem.current = params;
      setTimeout(()=>{
        setShow(isAllRanked(newList));
      }, 500)
    }
  }

  function submitRankingResult() {
    const groups = document.getElementsByClassName("dnd-group");
    const rankingResults = {"prompt":props.prompt,"rankings":[],"user":user?user:"Anonymous"};
    let count = 0;
    for(let i = props.numAnswers; i < props.numAnswers*2; i++) {
      const textareas=groups[i].querySelectorAll("textarea");
      const rankingResult = {"rank":i-props.numAnswers+1, "rankingAnswers":[]};
      if(textareas !== null){
        for(let j = 0; j < textareas.length; j++) {
          rankingResult["rankingAnswers"][j] = String(textareas[j].value)
          count += 1;
        }
        rankingResults["rankings"][i-props.numAnswers] = rankingResult
      }
    }
    // if(count === props.numAnswers) console.log(rankingResults["rankings"]);
    // else console.log("props.numAnswers",props.numAnswers,"count",count);

    // Send the ranking result to the backend
    axios.post('http://localhost:9990/submit_ranking/', 
               rankingResults ).then(res =>{
      const stateDuration = 1500;
      const pendingClassName = 'loading-btn--pending';
      const successClassName = 'loading-btn--success';
      const failClassName    = 'loading-btn--fail';
      const classNameToBeAdded = res.data.success?successClassName:failClassName;
      const elem = document.getElementsByClassName("button-wrapper")[0].querySelector("button");
      const p = document.getElementsByClassName("submitButton")[0].querySelector("p")
      elem.classList.add(pendingClassName);

      // button animation here
      window.setTimeout(() => {
          elem.classList.remove(pendingClassName);
          elem.classList.add(classNameToBeAdded);
        
          window.setTimeout(() => {
            elem.classList.remove(classNameToBeAdded);
            // If success clear the ranking data on web. Otherwise, show error message
            if(res.data.success) {
              props.updateData(constructDefaultPropsData());
              props.updatePrompt('')
            } else p.innerHTML = res.data.message
          }, stateDuration);
      }, stateDuration);
      
    })
  }

  function changeData(e, params) {
    const grpIdx = params.grpIdx, itemIdx = params.itemIdx
    let newList = JSON.parse(JSON.stringify(props.data));
    newList[grpIdx].items.splice(itemIdx, 1)
    newList[grpIdx].items.splice(itemIdx,0,e.target.value)
    props.updateData(newList)
  }

  const getStyle=(params) => {
    if(params.wrapIdx === dragItem.current.wrapIdx && params.grpIdx === dragItem.current.grpIdx && params.itemIdx === dragItem.current.itemIdx) return "current dnd-card"
    return "dnd-card"
  }

  const isAllRanked=(newList) => {
    for(let i = 0; i < props.numAnswers; i++){
      if(newList[i].items.length !== 0) return false;
    }
    return true;
  }

  return (
    <div className="DragNDropApp">
      <div className='dnd-title'>
          To be ranked
      </div>
      <div className='drag-n-drop'>
        
        {props.data.map((grp,grpIdx)=>(
          <div key={grpIdx} 
          onDragEnter={dragging && grp.items.length===0?(e) => onDragEnterHandle(e, {grpIdx, itemIdx:0}):null}
          className='dnd-group'
          >
            <div key={grp.group+grpIdx} className="group-title">
              {grpIdx < props.numAnswers?null:grp.group}
            </div>
            {grp.items.map((item, itemIdx)=>(
              <div key={itemIdx} 
                draggable 
                onDragStart={(e) => onDragStartHandle(e, {grpIdx, itemIdx})} 
                onDragEnter={(e) => onDragEnterHandle(e, {grpIdx, itemIdx})}
                className={dragging? getStyle({grpIdx, itemIdx}):"dnd-card"}
                id="dnd-card"
              >
                {item===undefined || item===null?null:<textarea value={item}
                className={dragging? getStyle({grpIdx, itemIdx}):"dnd-textarea"}
                id="dnd-textarea" 
                onChange={(e)=>changeData(e, {grpIdx, itemIdx})} >
                </textarea>}
                
              </div>
            ))}
          </div>
        ))}
      </div>
      <div className="submitButton button-wrapper">
        {<p></p>}
        {/* {show?<button onClick={submitRankingResult}>Submit</button>:null} */}
        {show?<span className="loading-btn-wrapper">
          <button className="loading-btn js_success-animation-trigger" onClick={submitRankingResult}>
            <span className="loading-btn__text">
              Submit
            </span>
          </button>
        </span>:null}
      </div>
    </div>
  )
}

export default DragNDrop;