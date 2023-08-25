import React from "react";
import styled from "styled-components";
import {motion} from "framer-motion"
function Home() {
  return (
    <motion.div className="ranking-container"
    initial={{opacity: 0}}
    animate={{opacity: 1}}
    transition={{delay:0.6, duration:0.6}}>
      <Container>
        <p>Welcome to use TACC GPT!</p>
      </Container>
    </motion.div>
  )
    
}

const Container = styled.main`
  position: relative;
  align-items: center;
  min-height: calc(100vh - 250px);
  overflow-x: hidden;
  display: block;
  padding: 0 calc(3.5vw + 5px);
  margin-top: 80px;
  text-align: center;
  font-size: 5rem;

  &:after {
    background-color: white;
    position: absolute;
    inset: 0px;
    z-index: -1;
  }
`;

export default Home;