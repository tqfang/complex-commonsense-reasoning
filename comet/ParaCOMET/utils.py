dimensions_of_interest = [
  "xNeed",
  "xIntent",
  "xWant",
  "oEffect",
  "xReact",
  "oWant",
  "oReact",
  "xEffect",
  "xAttr",
]

NL_to_rel = {
  'PersonX is likely:':"xEffect",
 'PersonX is seen as:':"xAttr",
 'PersonX needed:':"xNeed",
 'PersonX then feels:':"xReact",
 'PersonX wanted:':"xIntent",
 'PersonX wants:':"xWant",
 'PersonY/Others are likely:':"oEffect",
 'PersonY/Others then feel:':"oReact",
 'PersonY/Others want:':"oWant",
}

rel_to_question = {
  'xEffect':'What will PersonX do after',
  'xReact': 'What does PersonX feel after',
  'xWant': 'What does PersonX want to do after',
  'xAttr': 'What can PersonX be seen as given',
  'xNeed': 'What does PersonX need to do before',
  'xIntent': 'What does PersonX intent to do before',
  'oEffect': 'What will PersonY do after',
  'oWant':'What does PersonY want to do after',
  'oReact':'What does PersonY feel after',
}

relation_prompt = {
    "xIntent": "PersonX intent to do before",
    "xNeed": "PersonX needed to do before",
    "xWant": "PersonX wants to do after",
    "xEffect": "PersonX will do after",
    "xReact": "PersonX feels after",
    "xAttr": "PersonX is seen as given",
    "oEffect": "PersonY will do after",
    "oReact": "PersonY feels after",
    "oWant": "PersonY wants to do after",
    "HinderedBy": "hindered",
}