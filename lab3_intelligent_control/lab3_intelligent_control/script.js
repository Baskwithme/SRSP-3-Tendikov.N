const $=id=>document.getElementById(id),f2=(x,d=2)=>x.toFixed(d);
const secs=[['home','Титул и цель'],['proc','Схема процесса'],['model','Модель'],['pred','Прогнозирование'],['ctrl','Управление'],['qa','Вопросы'],['concl','Выводы']];
secs.forEach(([id,nm])=>{const b=document.createElement('button');b.textContent=nm;b.dataset.s=id;b.onclick=()=>show(id);$('nav').appendChild(b)});
function show(id){secs.forEach(([s])=>$(s).classList.toggle('on',s===id));[...$('nav').children].forEach(b=>b.classList.toggle('on',b.dataset.s===id));
 if(id==='model')runModel();if(id==='pred'&&!DS)train(42);if(id==='ctrl'){if(!DS)train(42);oneBatch();mc()}if(id==='concl'){if(!DS)train(42);mc()}scrollTo(0,0)}
const rng=s=>()=>{s|=0;s=s+0x6D2B79F5|0;let t=Math.imul(s^s>>>15,1|s);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
const gauss=r=>Math.sqrt(-2*Math.log(1-r()))*Math.cos(2*Math.PI*r());
const TGT=700,TOL=0.05;
const mu=(T,n,t,C)=>900*Math.exp(-0.035*(T-25))*Math.pow(C/50,2)*Math.pow(300/n,0.25)*(1+0.5*Math.exp(-t/8));
const PM=1.8e5,KL=150;
function step(T,Ts,n,Te){const h=Math.max(0,Math.min(1e4,2000*(Ts-T))),st=0.03*n*n;return{T:T+60*(h+0.5*st-KL*(T-Te))/PM,e:(h+st)/60000}}
const ch={};function mk(id,type,labels,ds,xt,yt,title){if(ch[id])ch[id].destroy();
 ch[id]=new Chart($(id),{type,data:{labels,datasets:ds},options:{responsive:true,maintainAspectRatio:false,animation:false,plugins:{title:{display:true,text:title,color:'#888'},legend:{labels:{color:'#888'}}},scales:{x:{title:{display:true,text:xt,color:'#888'},ticks:{color:'#888',maxTicksLimit:10}},y:{title:{display:true,text:yt,color:'#888'},ticks:{color:'#888'}}}}})}
const cols=['#d1495b','#00798c','#edae49','#6a994e','#7b4ea3'];
['mT','mn','mC'].forEach(i=>$(i).addEventListener('input',runModel));
function runModel(){const Ts=+$('mT').value,n=+$('mn').value,C=+$('mC').value;$('mTv').textContent=Ts;$('mnv').textContent=n;$('mCv').textContent=C;
 let T=22,E=0;const L=[],Tt=[],M=[];for(let t=0;t<=40;t++){L.push(t);Tt.push(T);M.push(mu(T,n,t,C));const s=step(T,Ts,n,20);T=s.T;E+=s.e}
 const m=M[40],ok=Math.abs(m-TGT)/TGT<=TOL;
 $('mres').innerHTML=`<div><b>${f2(Tt[40],1)} °C</b><span>T через 40 мин</span></div><div><b>${f2(m,0)} мПа·с</b><span>Вязкость μ</span></div><div><b>${f2(E)} кВт·ч</b><span>Энергия</span></div><div><b class="${ok?'ok':'no'}">${ok?'В допуске':'Вне допуска'}</b><span>Цель 700±5 %</span></div>`;
 mk('c1','line',L,[{label:'T, °C',data:Tt,borderColor:cols[0],pointRadius:0}],'Время t, мин','Температура T, °C','Тепловая модель: T(t)');
 mk('c2','line',L,[{label:'μ(t), мПа·с',data:M,borderColor:cols[1],pointRadius:0},{label:'Цель 700',data:L.map(()=>TGT),borderColor:'#888',borderDash:[5,4],pointRadius:0},{label:'+5 %',data:L.map(()=>TGT*1.05),borderColor:cols[2],borderDash:[2,3],pointRadius:0},{label:'−5 %',data:L.map(()=>TGT*.95),borderColor:cols[2],borderDash:[2,3],pointRadius:0}],'Время t, мин','Вязкость μ, мПа·с','Вязкость μ(t)')}
// ---------- ML ----------
let DS,TR,TE,LRw,RF,BEST,fx=d=>[1,d.T,Math.log(d.n),d.t,Math.log(d.C)];
function solve(A,b){const n=b.length;for(let i=0;i<n;i++){let p=i;for(let j=i+1;j<n;j++)if(Math.abs(A[j][i])>Math.abs(A[p][i]))p=j;[A[i],A[p]]=[A[p],A[i]];[b[i],b[p]]=[b[p],b[i]];for(let j=i+1;j<n;j++){const m=A[j][i]/A[i][i];for(let k=i;k<n;k++)A[j][k]-=m*A[i][k];b[j]-=m*b[i]}}const x=Array(n).fill(0);for(let i=n-1;i>=0;i--){let s=b[i];for(let k=i+1;k<n;k++)s-=A[i][k]*x[k];x[i]=s/A[i][i]}return x}
const sse=a=>{const m=a.reduce((s,d)=>s+d.y,0)/a.length;return a.reduce((s,d)=>s+(d.y-m)**2,0)};
function grow(D,dep,r){const m=D.reduce((s,d)=>s+d.y,0)/D.length;if(dep==0||D.length<6)return{v:m};let b=null;
 for(const f of ['T','n','t','C'].sort(()=>r()-.5).slice(0,3)){const v=D.map(d=>d[f]),lo=Math.min(...v),hi=Math.max(...v);for(let k=0;k<8;k++){const th=lo+(hi-lo)*r(),L=D.filter(d=>d[f]<=th),R=D.filter(d=>d[f]>th);if(L.length<3||R.length<3)continue;const s=sse(L)+sse(R);if(!b||s<b.s)b={s,f,th,L,R}}}
 return b?{f:b.f,th:b.th,l:grow(b.L,dep-1,r),r:grow(b.R,dep-1,r)}:{v:m}}
const tp=(t,d)=>t.v!==undefined?t.v:tp(d[t.f]<=t.th?t.l:t.r,d);
const pLR=d=>Math.exp(fx(d).reduce((s,v,i)=>s+v*LRw[i],0)),pRF=d=>RF.reduce((s,t)=>s+tp(t,d),0)/RF.length;
const pr=(T,n,t,C)=>{const d={T,n,t,C};return BEST==='RF'?pRF(d):pLR(d)};
function metr(P,A){const n=A.length,mA=A.reduce((a,b)=>a+b,0)/n;let ae=0,se=0,st=0,ap=0;A.forEach((a,i)=>{const e=P[i]-a;ae+=Math.abs(e);se+=e*e;st+=(a-mA)**2;ap+=Math.abs(e)/a});return{mae:ae/n,rmse:Math.sqrt(se/n),r2:1-se/st,mape:100*ap/n}}
const corr=(x,y)=>{const n=x.length,mx=x.reduce((a,b)=>a+b)/n,my=y.reduce((a,b)=>a+b)/n;let s=0,a=0,b=0;x.forEach((v,i)=>{s+=(v-mx)*(y[i]-my);a+=(v-mx)**2;b+=(y[i]-my)**2});return s/Math.sqrt(a*b)};
$('bre').onclick=()=>train(Math.floor(Math.random()*1e6));
function train(seed){const r=rng(seed);DS=[];for(let i=0;i<300;i++){const T=20+30*r(),n=100+400*r(),t=2+18*r(),C=40+20*r();DS.push({T,n,t,C,y:mu(T,n,t,C)*Math.exp(0.04*gauss(r))})}
 for(let i=299;i>0;i--){const j=Math.floor(r()*(i+1));[DS[i],DS[j]]=[DS[j],DS[i]]}
 TR=DS.slice(0,240);TE=DS.slice(240);
 const X=TR.map(fx),y=TR.map(d=>Math.log(d.y)),A=[...Array(5)].map((_,i)=>[...Array(5)].map((_,j)=>X.reduce((s,x)=>s+x[i]*x[j],0))),b=[...Array(5)].map((_,i)=>X.reduce((s,x,k)=>s+x[i]*y[k],0));LRw=solve(A,b);
 RF=[];for(let k=0;k<40;k++){RF.push(grow(TR.map(()=>TR[Math.floor(r()*TR.length)]),6,r))}
 const A_=TE.map(d=>d.y),mL=metr(TE.map(pLR),A_),mR=metr(TE.map(pRF),A_);BEST=mR.rmse<mL.rmse?'RF':'LR';
 const nm={T:'T, °C',n:'n, об/мин',t:'t, мин',C:'C, %'};let st='<tr><th>Признак</th><th>Мин</th><th>Среднее</th><th>Макс</th><th>Станд. откл.</th><th>Корреляция с μ</th></tr>';
 const yy=DS.map(d=>d.y);for(const k of ['T','n','t','C','y']){const v=DS.map(d=>d[k]),m=v.reduce((a,b)=>a+b)/v.length;st+=`<tr><td>${k==='y'?'μ, мПа·с':nm[k]}</td><td>${f2(Math.min(...v),1)}</td><td>${f2(m,1)}</td><td>${f2(Math.max(...v),1)}</td><td>${f2(Math.sqrt(v.reduce((s,x)=>s+(x-m)**2,0)/v.length),1)}</td><td>${k==='y'?'—':f2(corr(v,yy))}</td></tr>`}
 $('tst').innerHTML=st;
 const row=(n,m)=>`<tr><td>${n}</td><td>${f2(m.mae,1)}</td><td>${f2(m.rmse,1)}</td><td>${f2(m.r2,3)}</td><td>${f2(m.mape,1)}</td></tr>`;
 $('tm').innerHTML='<tr><th>Модель</th><th>MAE, мПа·с</th><th>RMSE, мПа·с</th><th>R²</th><th>MAPE, %</th></tr>'+row('Линейная регрессия (ln μ)',mL)+row('Случайный лес',mR);
 $('tbest').innerHTML=`Для управления выбрана модель с меньшей RMSE: <b>${BEST==='RF'?'случайный лес':'линейная регрессия'}</b>. Рекомендация: использовать прогноз как подсказку для коррекции режима с контролем границ и периодическим переобучением на реальных данных.`;
 let ts='<tr><th>№</th><th>T</th><th>n</th><th>t</th><th>C</th><th>μ факт</th><th>μ регрессия</th><th>μ лес</th></tr>';TE.slice(0,8).forEach((d,i)=>ts+=`<tr><td>${i+1}</td><td>${f2(d.T,1)}</td><td>${f2(d.n,0)}</td><td>${f2(d.t,1)}</td><td>${f2(d.C,1)}</td><td>${f2(d.y,0)}</td><td>${f2(pLR(d),0)}</td><td>${f2(pRF(d),0)}</td></tr>`);$('ts').innerHTML=ts;
 mk('c3','scatter',undefined,[{label:'Линейная регрессия',data:TE.map(d=>({x:d.y,y:pLR(d)})),backgroundColor:cols[1]},{label:'Случайный лес',data:TE.map(d=>({x:d.y,y:pRF(d)})),backgroundColor:cols[0]},{label:'Идеал y = x',type:'line',data:[{x:300,y:300},{x:1700,y:1700}],borderColor:'#888',borderDash:[5,4],pointRadius:0}],'Фактическая вязкость, мПа·с','Прогноз вязкости, мПа·с','Факт и прогноз на тестовой выборке');
 MC=null}
// ---------- control ----------
let MC=null;
function batch(smart,seed){const r=rng(seed),Ct=50+2.5*gauss(r),Te=20+3*gauss(r),lim=smart?40:30;let T=22,Ts=35,n=300,E=0,ok=0,hit=null,fin=lim,es=0,en=0,dv=0,dn=0;const L=[],M=[];
 for(let t=1;t<=lim;t++){const s=step(T,Ts,n,Te);T=s.T;E+=s.e;const m=mu(T,n,t,Ct);L.push(t);M.push(m);if(t>=5){dv+=Math.abs(T-Ts);dn++}
  if(hit===null&&Math.abs(m-TGT)/TGT<=TOL)hit=t;
  if(smart){const mh=pr(T,n,t,Ct*(1+0.02*gauss(r))),e=(mh-TGT)/TGT;es+=Math.abs(mh-m)/m;en++;if(Math.abs(e)<=TOL*0.8){if(++ok>=2){fin=t;break}}else ok=0;Ts=Math.min(50,Math.max(25,Ts+25*e));n=Math.min(500,Math.max(100,n+200*e))}}
 const m=M[M.length-1];return{L,M,E,hit:hit===null?lim:hit,dev:dn?dv/dn:0,err:en?100*es/en:null,fin:smart?fin:30,ok:Math.abs(m-TGT)/TGT<=TOL,mu:m}}
$('bb').onclick=oneBatch;
function oneBatch(){const s=Math.floor(Math.random()*1e6),a=batch(false,s),b=batch(true,s),L=[...Array(40)].map((_,i)=>i+1);
 const pad=x=>L.map((_,i)=>x[i]??null);
 $('cres').innerHTML=`<div><b>${f2(a.mu,0)}</b><span>μ традиц., мПа·с (${a.ok?'в допуске':'вне допуска'})</span></div><div><b>${f2(b.mu,0)}</b><span>μ интеллект., мПа·с (${b.ok?'в допуске':'вне допуска'})</span></div><div><b>${a.fin} / ${b.fin} мин</b><span>Длительность партии</span></div><div><b>${f2(a.E)} / ${f2(b.E)}</b><span>Энергия, кВт·ч</span></div>`;
 mk('c4','line',L,[{label:'Традиционное',data:pad(a.M),borderColor:cols[0],pointRadius:0},{label:'Интеллектуальное',data:pad(b.M),borderColor:cols[1],pointRadius:0},{label:'Цель 700',data:L.map(()=>TGT),borderColor:'#888',borderDash:[5,4],pointRadius:0},{label:'±5 %',data:L.map(()=>TGT*1.05),borderColor:cols[2],borderDash:[2,3],pointRadius:0},{label:'−5 %',data:L.map(()=>TGT*.95),borderColor:cols[2],borderDash:[2,3],pointRadius:0}],'Время t, мин','Вязкость μ, мПа·с','Одна партия: вязкость при двух подходах')}
function mc(){if(MC)return;const N=200,R={a:[],b:[]};for(let i=0;i<N;i++){R.a.push(batch(false,1000+i));R.b.push(batch(true,1000+i))}
 const av=(x,k)=>x.reduce((s,d)=>s+d[k],0)/x.length,pc=x=>100*x.filter(d=>d.ok).length/x.length;
 MC={ta:av(R.a,'hit'),tb:av(R.b,'hit'),Ea:av(R.a,'E'),Eb:av(R.b,'E'),da:av(R.a,'dev'),db:av(R.b,'dev'),pa:pc(R.a),pb:pc(R.b),er:av(R.b,'err'),fb:av(R.b,'fin')};
 const q=(n,a,b)=>`<tr><td>${n}</td><td>${a}</td><td>${b}</td></tr>`;
 $('cmp').innerHTML='<tr><th>Критерий</th><th>Традиционное</th><th>Интеллектуальное</th></tr>'+q('Ошибка прогноза вязкости, %','—',f2(MC.er,1))+q('Откл. T от уставки (после 5 мин), °C',f2(MC.da),f2(MC.db))+q('Время достижения допуска, мин',f2(MC.ta,1),f2(MC.tb,1))+q('Длительность партии, мин','30',f2(MC.fb,1))+q('Энергопотребление, кВт·ч',f2(MC.Ea),f2(MC.Eb))+q('Доля партий в допуске, %',f2(MC.pa,1),f2(MC.pb,1));
 $('cmpt').innerHTML='<b>Преимущества интеллектуального:</b> компенсирует разброс состава сырья, повышает долю годной продукции, сокращает время и энергию. Большое отклонение T от уставки у интеллектуального подхода объясняется тем, что сама уставка активно меняется, а нагрев инерционен. <b>Ограничения:</b> зависит от качества датчиков и обучающих данных, требует переобучения и защиты границ режима; традиционное проще, надёжнее и дешевле во внедрении.';
 $('cc').innerHTML=`<ul><li>Построены схема процесса, тепловая модель и модель вязкости μ(T, n, t, C).</li><li>Лучшая прогнозная модель (${BEST==='RF'?'случайный лес':'линейная регрессия'}) дала R² на тесте выше 0,9 при простом признаковом описании. Логарифмирование делает регрессию точной для мультипликативной зависимости, а лес на 240 записях аппроксимирует её ступенчато.</li><li>В 200 моделируемых партиях доля годных: ${f2(MC.pa,1)} % (традиционное) против ${f2(MC.pb,1)} % (интеллектуальное); энергия ${f2(MC.Ea)} против ${f2(MC.Eb)} кВт·ч.</li><li>Интеллектуальное управление выгодно при вариациях сырья, но требует датчиков, данных и защиты от выхода за ограничения оборудования.</li></ul>`}
const QA=[['Основные этапы приготовления ЛКМ?','Подача и дозирование компонентов, загрузка, перемешивание и диспергирование пигментов, регулирование вязкости и температуры, контроль качества, выдача готового продукта.'],['Какие параметры влияют на качество?','Температура, частота вращения мешалки, время смешивания, соотношение компонентов (концентрация), вязкость сырья.'],['Зачем математическое моделирование?','Чтобы прогнозировать поведение процесса, подбирать режимы без дорогих опытов, оптимизировать энергозатраты и проектировать управление.'],['Какие методы ML для прогноза вязкости?','Линейная регрессия, дерево решений, случайный лес, нейронная сеть.'],['Чем отличается традиционное и интеллектуальное управление?','Традиционное держит заданные уставки и реагирует на отклонения; интеллектуальное использует прогнозную модель и корректирует режим заранее с учётом состояния процесса.'],['Какие данные нужны для обучения?','Журналы T, n, t, составов и измеренной вязкости (лаборатория/вискозиметр), по разным рецептурам и условиям, очищенные от выбросов.'],['Как оценивается точность прогноза?','На тестовой выборке: MAE, RMSE, MAPE, R²; плюс кросс-валидация и сравнение с лабораторным замером.'],['Функции датчиков?','Измеряют T, μ, n, мощность, уровень; дают обратную связь контроллеру и позволяют проверять корректность данных.'],['Как снизить энергопотребление при сохранении качества?','Завершать смешивание по достижении допуска, выбирать минимальную достаточную скорость (E ~ n²), использовать прогноз вместо запаса по времени.'],['Ограничения внедрения?','Пределы оборудования (мощность, n, T), рецептура и регламент, надёжность датчиков, качество данных, безопасность, необходимость валидации и переобучения.']];
$('qlist').innerHTML=QA.map((q,i)=>`<details><summary>${i+1}. ${q[0]}</summary><p>${q[1]}</p></details>`).join('');
show('home');
