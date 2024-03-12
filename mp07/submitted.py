'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''

import copy, queue
import pprint


# 这个 mp 真尼玛抽象
def standardize_variables(nonstandard_rules):
    '''
    @param nonstandard_rules (dict) - dict from ruleIDs to rules
        Each rule is a dict:
        rule['antecedents'] contains the rule antecedents (a list of propositions)
        rule['consequent'] contains the rule consequent (a proposition).
   
    @return standardized_rules (dict) - an exact copy of nonstandard_rules,
        except that the antecedents and consequent of every rule have been changed
        to replace the word "something" with some variable name that is
        unique to the rule, and not shared by any other rule.
    @return variables (list) - a list of the variable names that were created.
        This list should contain only the variables that were used in rules.
    '''
    # raise RuntimeError("You need to write this part!")
    standardized_rules = copy.deepcopy(nonstandard_rules)
    variables = []
    for key in standardized_rules:
      if 'rule' in key: # pattern check
        # since keys are guarantee to be unique, so they are the good choices
        variables.append(key)   
        for ante in standardized_rules[key]['antecedents']:
          for i in range(len(ante)):
            if ante[i] == 'something':
              ante[i] = key
        for cons in range(len(standardized_rules[key]['consequent'])):
          if standardized_rules[key]['consequent'][cons] == 'something':
            standardized_rules[key]['consequent'][cons] = key         

    return standardized_rules, variables

def unify(query, datum, variables):
    '''
    @param query: proposition that you're trying to match.
      The input query should not be modified by this function; consider deepcopy.
    @param datum: proposition against which you're trying to match the query.
      The input datum should not be modified by this function; consider deepcopy.
    @param variables: list of strings that should be considered variables.
      All other strings should be considered constants.
    
    Unification succeeds if (1) every variable x in the unified query is replaced by a 
    variable or constant from datum, which we call subs[x], and (2) for any variable y
    in datum that matches to a constant in query, which we call subs[y], then every 
    instance of y in the unified query should be replaced by subs[y].

    @return unification (list): unified query, or None if unification fails.
    @return subs (dict): mapping from variables to values, or None if unification fails.
       If unification is possible, then answer already has all copies of x replaced by
       subs[x], thus the only reason to return subs is to help the calling function
       to update other rules so that they obey the same substitutions.

    Examples:

    unify(['x', 'eats', 'y', False], ['a', 'eats', 'b', False], ['x','y','a','b'])
      unification = [ 'a', 'eats', 'b', False ]
      subs = { "x":"a", "y":"b" }
    unify(['bobcat','eats','y',True],['a','eats','squirrel',True], ['x','y','a','b'])
      unification = ['bobcat','eats','squirrel',True]
      subs = { 'a':'bobcat', 'y':'squirrel' }
    unify(['x','eats','x',True],['a','eats','a',True],['x','y','a','b'])
      unification = ['a','eats','a',True]
      subs = { 'x':'a' }
    unify(['x','eats','x',True],['a','eats','bobcat',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'x':'a', 'a':'bobcat'}
      When the 'x':'a' substitution is detected, the query is changed to 
      ['a','eats','a',True].  Then, later, when the 'a':'bobcat' substitution is 
      detected, the query is changed to ['bobcat','eats','bobcat',True], which 
      is the value returned as the answer.
    unify(['a','eats','bobcat',True],['x','eats','x',True],['x','y','a','b'])
      unification = ['bobcat','eats','bobcat',True],
      subs = {'a':'x', 'x':'bobcat'} 
      When the 'a':'x' substitution is detected, the query is changed to 
      ['x','eats','bobcat',True].  Then, later, when the 'x':'bobcat' substitution 
      is detected, the query is changed to ['bobcat','eats','bobcat',True], which is 
      the value returned as the answer.
    unify([...,True],[...,False],[...]) should always return None, None, regardless of the 
      rest of the contents of the query or datum.
    '''
    # raise RuntimeError("You need to write this part!")

    
    if query[-1] != datum[-1]:
      return None, None
    
    unification = []  # or None if something fails
    subs = {}         # or None if something fails
    # query_copy = copy.deepcopy(query)
    # datum_copy = copy.deepcopy(datum)
    

    # 这个有点绕，tmd
    for i in range(len(query)):
      
      # track one to one mapping in query
      if query[i] in variables:
        if query[i] not in subs or subs[query[i]] == datum[i]: # keep searching until an variable (or a word) is reached
          subs[query[i]] = datum[i] 
        else:
          temp = subs[query[i]]
          while temp in subs: temp = subs[temp]
          if temp in variables:
            subs[temp] = datum[i]
          else:
            return None, None
          
      # track one to one mapping in datum
      elif datum[i] in variables:
        if datum[i] not in subs or subs[datum[i]] == query[i]:
          subs[datum[i]] = query[i]
        else:
          temp = subs[datum[i]]
          while temp in subs: temp = subs[temp] # keep searching until an variable (or a word) is reached
          if temp in variables:
            subs[temp] = query[i]
          else:
            return None, None

      else:
        if query[i] != datum[i]:
          return None, None
         
        
      

    # translate unification
    for word in query:
      if word not in variables:
        unification.append(word)
      else:
        while word in subs:
          word = subs[word]
        unification.append(word)



    
  
    
    return unification, subs

def apply(rule, goals, variables):
    '''
    @param rule: A rule that is being tested to see if it can be applied
      This function should not modify rule; consider deepcopy.
    @param goals: A list of propositions against which the rule's consequent will be tested
      This function should not modify goals; consider deepcopy.
    @param variables: list of strings that should be treated as variables

    Rule application succeeds if the rule's consequent can be unified with any one of the goals.
    
    @return applications: a list, possibly empty, of the rule applications that
       are possible against the present set of goals.
       Each rule application is a copy of the rule, but with both the antecedents 
       and the consequent modified using the variable substitutions that were
       necessary to unify it to one of the goals. Note that this might require 
       multiple sequential substitutions, e.g., converting ('x','eats','squirrel',False)
       based on subs=={'x':'a', 'a':'bobcat'} yields ('bobcat','eats','squirrel',False).
       The length of the applications list is 0 <= len(applications) <= len(goals).  
       If every one of the goals can be unified with the rule consequent, then 
       len(applications)==len(goals); if none of them can, then len(applications)=0.
    @return goalsets: a list of lists of new goals, where len(newgoals)==len(applications).
       goalsets[i] is a copy of goals (a list) in which the goal that unified with 
       applications[i]['consequent'] has been removed, and replaced by 
       the members of applications[i]['antecedents'].

    Example:
    rule={
      'antecedents':[['x','is','nice',True],['x','is','hungry',False]],
      'consequent':['x','eats','squirrel',False]
    }
    goals=[
      ['bobcat','eats','squirrel',False],
      ['bobcat','visits','squirrel',True],
      ['bald eagle','eats','squirrel',False]
    ]
    variables=['x','y','a','b']

    applications, newgoals = submitted.apply(rule, goals, variables)

    applications==[
      {
        'antecedents':[['bobcat','is','nice',True],['bobcat','is','hungry',False]],
        'consequent':['bobcat','eats','squirrel',False]
      },
      {
        'antecedents':[['bald eagle','is','nice',True],['bald eagle','is','hungry',False]],
        'consequent':['bald eagle','eats','squirrel',False]
      }
    ]
    newgoals==[
      [
        ['bobcat','visits','squirrel',True],
        ['bald eagle','eats','squirrel',False]
        ['bobcat','is','nice',True],
        ['bobcat','is','hungry',False]
      ],[
        ['bobcat','eats','squirrel',False]
        ['bobcat','visits','squirrel',True],
        ['bald eagle','is','nice',True],
        ['bald eagle','is','hungry',False]
      ]
    '''
    # raise RuntimeError("You need to write this part!")
    applications = []
    goalsets = []
   
    for g in goals:
      goals_cpy = copy.deepcopy(goals)
      rule_cpy = copy.deepcopy(rule)
      uni, subs = unify(rule_cpy['consequent'], g, variables) # apply unification to consequent
      if uni != None:
        rule_cpy['consequent'] = uni
        goals_cpy.remove(g)
        for i in range(len(rule_cpy['antecedents'])): # use the substitution to antecedents if the unification is good
          for j in range(len(rule_cpy['antecedents'][i])): # subs all the words 
            word = rule_cpy['antecedents'][i][j]
            while  word in subs:            # keep looking the word
              word = subs[word]
            rule_cpy['antecedents'][i][j] = word
          goals_cpy.append(rule_cpy['antecedents'][i])         # set antecedents as the new goal
        applications.append(rule_cpy)      # add the modified rule to application
        goalsets.append(goals_cpy)
      else:
        continue

    return applications, goalsets

def backward_chain(query, rules, variables):
    '''
    @param query: a proposition, you want to know if it is true
    @param rules: dict mapping from ruleIDs to rules
    @param variables: list of strings that should be treated as variables
 
    @return proof (list): a list of rule applications
      that, when read in sequence, conclude by proving the truth of the query.
      If no proof of the query was found, you should return proof=None.
    '''
    # raise RuntimeError("You need to write this part!")
    proof = []
    out_bound_proof = []
    rule_path = dict()
    rule_set = dict()
    bfs_q = queue.Queue()
    bfs_q.put([query]) # put the first goal set to the queue
    rule_path[''.join(map(str, [query]))] = 0

    while not bfs_q.empty():    
      goal_set = bfs_q.get()  # getting a set of goals from the queue
      if len(goal_set) == 0: break  # goal is empty then we found it
      for each_rule in rules: # apply each rule to the goal set
        new_apply, new_goals = apply(rules[each_rule], goal_set, variables)
        if len(new_apply) > 0:
          for app in new_apply: out_bound_proof.insert(0, app)
          for g in new_goals:
            state = ''.join(map(str,g)) # convert it to string 

            rule_path[state] = ''.join(map(str, goal_set)) # the previous node is the goal_set
            rule_set[state] = new_apply

            bfs_q.put(g)  # putting goal list to the place
    
    final = ''
    while final in rule_path and rule_path[final] != 0:
      for r in rule_set[final]:
        proof.append(r)
      final = rule_path[final]
    
    if(len(proof) > 0):
      return proof
    if(len(out_bound_proof) > 0):
      return out_bound_proof
  
    return None
