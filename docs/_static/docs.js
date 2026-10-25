(function(){
  var KEY='jsl-docs-theme';
  var root=document.documentElement;
  var media=window.matchMedia?window.matchMedia('(prefers-color-scheme: dark)'):null;
  function resolve(pref){return pref==='auto'?(media&&media.matches?'dark':'light'):pref;}
  function apply(pref){
    root.setAttribute('data-theme',resolve(pref));
    root.setAttribute('data-theme-pref',pref);
    var sel=document.getElementById('theme-select');
    if(sel&&sel.value!==pref){sel.value=pref;}
    try{localStorage.setItem(KEY,pref);}catch(e){}
  }
  var pref='auto';
  try{pref=localStorage.getItem(KEY)||'auto';}catch(e){}
  var qm=/[?&]theme=(light|dark|auto)/.exec(location.search);
  if(qm){pref=qm[1];}
  apply(pref);
  var sel=document.getElementById('theme-select');
  if(sel){sel.addEventListener('change',function(){apply(sel.value);});}
  if(media&&media.addEventListener){media.addEventListener('change',function(){
    if((root.getAttribute('data-theme-pref')||'auto')==='auto'){apply('auto');}
  });}

  var mask=document.getElementById('search-mask');
  var dlg=document.getElementById('search-dialog');
  var input=document.getElementById('search-input');
  var box=document.getElementById('search-results');
  var data=window.__SEARCH_INDEX__||[];
  var ROOT=window.__DOCS_ROOT__||'';
  function closeSearch(){if(dlg){dlg.classList.remove('open');}if(mask){mask.classList.remove('open');}}
  function openSearch(){
    if(!dlg){return;}
    dlg.classList.add('open');if(mask){mask.classList.add('open');}
    input.value='';input.focus();box.innerHTML='';
  }
  function render(q){
    q=q.trim().toLowerCase();
    if(!q){box.innerHTML='';return;}
    var out=[];
    for(var i=0;i<data.length&&out.length<50;i++){
      var it=data[i];
      if(it[0].toLowerCase().indexOf(q)>=0||it[1].toLowerCase().indexOf(q)>=0||
         (it[3]&&it[3].toLowerCase().indexOf(q)>=0)){out.push(it);}
    }
    out.sort(function(a,b){return (a[5]==='type'?0:1)-(b[5]==='type'?0:1);});
    if(!out.length){box.innerHTML='<div class="search-empty">没有匹配项</div>';return;}
    var h='';
    for(var j=0;j<out.length;j++){
      var o=out[j];
      h+='<a href="'+ROOT+o[4]+'"><span class="sr-name">'+o[1]+'</span>'+
         '<span class="sr-meta">'+o[2]+(o[3]?' · '+o[3]:'')+'</span></a>';
    }
    box.innerHTML=h;
  }
  if(input){input.addEventListener('input',function(){render(input.value);});}
  document.addEventListener('keydown',function(e){
    if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openSearch();}
    else if(e.key==='Escape'){closeSearch();}
  });
  var trigger=document.getElementById('search-trigger');
  if(trigger){trigger.addEventListener('click',openSearch);}
  if(mask){mask.addEventListener('click',closeSearch);}
  if(input){input.addEventListener('keydown',function(e){
    if(e.key==='Enter'){var a=box.querySelector('a');if(a){location.href=a.getAttribute('href');}}
  });}
})();
