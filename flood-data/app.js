'use strict';
const search=document.querySelector('#source-search');
const category=document.querySelector('#category');
const cards=[...document.querySelectorAll('.source-card')];
const status=document.querySelector('#result-status');
const empty=document.querySelector('#empty');
function filterSources(){const term=search.value.trim().toLowerCase();let shown=0;cards.forEach(card=>{const show=(!term||card.textContent.toLowerCase().includes(term))&&(category.value==='all'||card.dataset.category===category.value);card.hidden=!show;if(show)shown++});status.textContent=`Showing ${shown} of ${cards.length} sources`;empty.hidden=shown!==0;}
search.addEventListener('input',filterSources);category.addEventListener('change',filterSources);document.querySelector('#reset').addEventListener('click',()=>{search.value='';category.value='all';filterSources();search.focus()});filterSources();
const candidateSearch=document.querySelector('#candidate-search');
if(candidateSearch){
 const candidates=[...document.querySelectorAll('.candidate')];
 const groups=[...document.querySelectorAll('.candidate-group')];
 const candidateStatus=document.querySelector('#candidate-result-status');
 const candidateEmpty=document.querySelector('#candidate-empty');
 function filterCandidates(){const term=candidateSearch.value.trim().toLowerCase();let shown=0;candidates.forEach(card=>{card.hidden=!!term&&!card.textContent.toLowerCase().includes(term);if(!card.hidden)shown++});groups.forEach(group=>{group.hidden=![...group.querySelectorAll('.candidate')].some(card=>!card.hidden)});candidateStatus.textContent=`Showing ${shown} of ${candidates.length} additional candidates`;candidateEmpty.hidden=shown!==0;}
 candidateSearch.addEventListener('input',filterCandidates);
 document.querySelector('#candidate-reset').addEventListener('click',()=>{candidateSearch.value='';filterCandidates();candidateSearch.focus()});filterCandidates();
}
