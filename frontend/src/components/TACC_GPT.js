import { useEffect } from "react";
function TACC_GPT() {
  useEffect(() => {
    window.location.replace('http://localhost:9995/TACC_GPT');
  }, [])
}

export default TACC_GPT;