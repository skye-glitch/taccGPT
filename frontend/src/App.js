import React from 'react';
import './App.css';
import {BrowserRouter as Router, Routes, Route} from "react-router-dom"


import Ranking from './components/Ranking';
import Home from './components/Home'
import Header from './components/Header';
import TACC_GPT from './components/TACC_GPT';
// import Database from './components/Database';
import ShowQAParis from './components/ShowQAPairs';
import ShowRankings from './components/ShowRankings';

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

          {/* <Route exact path='/database'>
            <Route path='/database' element={<Database />} />
          </Route> */}

          <Route exact path='/show_database_qa_pairs'>
            <Route path='/show_database_qa_pairs' element={<ShowQAParis />}/>
          </Route>

          <Route exact path='/show_database_rankings'>
            <Route path='/show_database_rankings' element={<ShowRankings />}/>
          </Route>

        </Routes>
      </Router>
      
    </div>
  );
}

export default App;
