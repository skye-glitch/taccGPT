import React, {useState} from 'react';
import './App.css';
import {BrowserRouter as Router, Routes, Route} from "react-router-dom"


import Ranking from './components/Ranking';
import Home from './components/Home'
import Header from './components/Header';
import TACC_GPT from './components/TACC_GPT';

function App() {
  
  return (
    <div className="App">
      <Router>
        <Header />
        <Routes>
          <Route exact path='/'>
            <Route path='/' element={<Home />}></Route>
          </Route>

          <Route exact path='/ranking'>
            <Route path='/ranking' element={<Ranking />}></Route>
          </Route>

          <Route path='/TACC_GPT' element={<TACC_GPT />} />
        </Routes>
      </Router>
      
    </div>
  );
}

export default App;
