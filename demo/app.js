"use strict";
const $ = (id) => document.getElementById(id);
const names = {q_learning:["Aprender a decidir","El agente elige movimientos usando su tabla Q. Mantiene un objetivo hasta capturarlo y después selecciona otro. No calcula rutas con BFS."],untrained_q:["Antes de aprender","La tabla Q está vacía: todos los movimientos legales tienen el mismo valor. El agente desempata al azar, sin actualizar la tabla."],random:["Explorar al azar","Elige al azar uno de los movimientos legales. Ve los mismos fantasmas, pero no usa esas posiciones para planificar ni aprende de las recompensas."],greedy_bfs:["Calcular una ruta","Busca el fantasma más cercano por los caminos del mapa con BFS. Esta referencia conoce las paredes y calcula rutas sin entrenamiento."]};
let current, frame=0, timer=null;
function pause(){if(timer!==null)clearInterval(timer);timer=null;$("play").textContent="Reproducir";}
function render(){
  const data=current.frames[frame], rows=current.board, size=32;
  const height=rows.length,width=rows[0].length;
  let content='<title>Movimiento '+frame+' de '+(current.frames.length-1)+'</title>';
  rows.forEach((row,y)=>[...row].forEach((cell,x)=>{if(cell==="%")content+=`<rect x="${x*size+4}" y="${y*size+4}" width="24" height="24" rx="5" fill="#243d61" stroke="#3c5c87"/>`;}));
  const px=data.pacman[0]*size+16,py=(height-1-data.pacman[1])*size+16;
  content+=`<path d="M${px} ${py} L${px+10.8} ${py-6.8} A13 13 0 1 0 ${px+10.8} ${py+6.8} Z" fill="#ffe66b"/>`;
  data.ghosts.forEach(([x,y],i)=>{if(y===1)return;const gx=x*size+16,gy=(height-1-y)*size+16;content+=`<path d="M${gx-11} ${gy+11}V${gy}a11 11 0 0 1 22 0v11l-5-4-6 4-6-4Z" fill="${i===0?'#fb8299':'#78c6f2'}"/><circle cx="${gx-4}" cy="${gy-1}" r="3" fill="white"/><circle cx="${gx+4}" cy="${gy-1}" r="3" fill="white"/>`;});
  $("board").setAttribute("viewBox",`0 0 ${width*size} ${height*size}`);$("board").innerHTML=content;
  $("step").textContent=`${frame} / ${current.frames.length-1}`;$("timeline").value=frame;
  const captured=data.ghosts.filter(([,y])=>y===1).length;
  $("outcome").textContent=frame===current.frames.length-1?(current.win?`Completada · ${current.moves} movimientos · ${data.score} puntos`:`Límite alcanzado · ${captured} de 2 fantasmas capturados`):`${captured} de 2 fantasmas capturados · ${data.score} puntos`;
}
function load(){pause();frame=0;current=window.REPLAYS[$("maze").value][$("agent").value];$("timeline").max=current.frames.length-1;const [title,description]=names[$("agent").value];$("agent-title").textContent=title;$("explanation").textContent=description;render();}
function start(){if(frame===current.frames.length-1)frame=0;$("play").textContent="Pausar";render();timer=setInterval(()=>{frame++;render();if(frame===current.frames.length-1)pause();},Number($("speed").value));}
$("play").addEventListener("click",()=>timer===null?start():pause());
$("reset").addEventListener("click",()=>{pause();frame=0;render();});
$("timeline").addEventListener("input",()=>{pause();frame=Number($("timeline").value);render();});
$("agent").addEventListener("change",load);$("maze").addEventListener("change",load);
$("speed").addEventListener("change",()=>{if(timer!==null){pause();start();}});
load();
