import styled from "styled-components";
import { useNavigate } from "react-router-dom";
import DropDown from "./DatabaseDropDown.tsx"
import * as Ariakit from "@ariakit/react";
const Header = (props) => {
  const navigate = useNavigate();
  
  return (
    <Nav>
      <Logo href="https://tacc.utexas.edu">
        <img src="/images/TACC-logo.svg" alt="Logo" />
      </Logo>
      <NavMenu>
        <a onClick={()=>{navigate("/")}}>
          <Ariakit.Button className="button">
          <img src="/images/home-1-svgrepo-com.svg" alt="HOME" />
            HOME
          </Ariakit.Button>
        </a>

        <a onClick={() => navigate("/TACC_GPT")}>
          <Ariakit.Button className="button">
          <img src="/images/robot-svgrepo-com.svg" alt="TACC GPT" />
            TACC GPT
          </Ariakit.Button>
        </a>

        <a onClick={() => navigate("/Ranking")}>
          <Ariakit.Button className="button">
            <img src="/images/ranking-icon.svg" alt="RANKING" />
            Ranking
          </Ariakit.Button>
        </a>

        <a>
          <DropDown />
        </a>
      </NavMenu>
      {/* {!userName? (<Login onClick={() => {navigate("/Login")}}>Login</Login>):(
        <SignOut>
          <UserImg src={userPhoto} alt={userName} />
          <DropDown>
            <span onClick={()=>navigate("/Dashboard")}>Dashboard</span>
            <br />
            <br />
            <span onClick={signout}>Sign out</span>    
          </DropDown>
        </SignOut>
      )} */}
    </Nav>
  );
};

const Nav = styled.nav`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 75px;
  background-color: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 24px;
  letter-spacing: 30px;
  z-index: 3;
`;

const Logo = styled.a`
  padding: 0;
  width: 80px;
  margin-top: 4px;
  max-height: 70px;
  font-size: 0;
  display: inline-block;

  img {
    display: block;
    width: 140%;
  }
`;

const NavMenu = styled.div`
  align-items: center;
  display: flex;
  flex-flow: row nowrap;
  height: 100%;
  justify-content: flex-end;
  margin: 0px;
  padding: 0px;
  position: relative;
  margin-right: auto;
  margin-left: 25px;

  a {
    display: flex;
    align-items: center;
    padding: 0 12px;

    img {
      height: 20px;
      min-width: 20px;
      width: 25px;
      z-index: auto;
    }

    span {
      color: black;
      font-size: 13px;
      letter-spacing: 1.42px;
      line-height: 1.08;
      padding: 2px 0px;
      white-space: nowrap;
      position: relative;

      &:before {
        background-color: black;
        border-radius: 0px 0px 4px 4px;
        bottom: -6px;
        content: "";
        height: 2px;
        left: 0px;
        opacity: 0;
        position: absolute;
        right: 0px;
        transform-origin: left center;
        transform: scaleX(0);
        transition: all 250ms cubic-bezier(0.25, 0.46, 0.45, 0.94) 0s;
        visibility: hidden;
        width: auto;
      }
    }

    &:hover {
      span:before {
        transform: scaleX(1);
        visibility: visible;
        opacity: 1 !important;
      }
    }
  }

  /* @media (max-width: 768px) {
    display: none;
  } */
`;

// const Login = styled.a`
//   background-color: rgba(0, 0, 0, 0.6);
//   padding: 8px 16px;
//   text-transform: uppercase;
//   letter-spacing: 1.5px;
//   border: 1px solid #f9f9f9;
//   border-radius: 4px;
//   transition: all 0.2s ease 0s;

//   &:hover {
//     background-color: #f9f9f9;
//     color: #000;
//     border-color: transparent;
//   }
// `;

// const UserImg = styled.img`
//   height: 200%;
// `;

// const DropDown = styled.div`
//   position: absolute;
//   top: 48px;
//   right: 0px;
//   background: rgb(19, 19, 19);
//   border: 1px solid rgba(151, 151, 151, 0.34);
//   border-radius: 4px;
//   box-shadow: rgb(0 0 0 / 50%) 0px 0px 18px 0px;
//   padding: 10px;
//   font-size: 14px;
//   letter-spacing: 3px;
//   width: 200px;
//   opacity: 0;
// `;

// const SignOut = styled.div`
//   position: relative;
//   height: 48px;
//   width: 48px;
//   display: flex;
//   cursor: pointer;
//   align-items: center;
//   justify-content: center;

//   ${UserImg} {
//     border-radius: 50%;
//     width: 100%;
//     height: 100%;
//   }

//   &:hover {
//     ${DropDown} {
//       opacity: 1;
//       transition-duration: 1s;
//     }
//   }
// `;

export default Header;