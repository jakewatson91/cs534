# HW 4
For this homework you are going to be given a STRIPS problem
but we are using a notation called PDDL and using a
Python library that supports planning with PDDL. 

# Step 0
```
pip install -r requirements.txt
```
# Step 1
The knowledge base and proposition syntax are all expressed 
in parentheses enclosures. 

For example, if you look at `problems/task01.pddl` you will see

```
(define (problem BLOCKS-4-0)
(:domain BLOCKS)
(:objects L - block)
(:INIT (position-1 L))
(:goal (position-4 L))
)
```

This defines the problem the planner needs to solve.
The problem predicate is the problem name. The domain
predicate defines the "knowledge base" we discussed and 
has all the actions defined. The INIT predicate defines the initial state
of our closed world and the goal predicate defines the end state we want to 
reach through some series of actions (a.k.a., our plan).

The actions we use to solve the above problem are defined in the `knowledge_base/domain.pddl` file.

```
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Op-block-L world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain BLOCKS)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (position-0 ?x)
	       (position-1 ?x)
	       (position-2 ?x)
	       (position-3 ?x)
	       (position-4 ?x)
	       )

  (:action rotate-1
	     :parameters (?x - block)
	     :precondition (position-0 ?x)
	     :effect
	     (and (not (position-0 ?x))
		   (position-1 ?x))
		   )

...

```

The snippet above shows how to create your actions and possible predicates in the
world you are creating. The domain predicate is the name of your "world". For the 
requirements predicate, this is specific to this library, and please put `(:requirements :strips :typing)`
as shown above. The types predicate lets you type your objects and here we are using a type we created called `blocks`.
The predicate "predicates" is where you define all possible predicates in your world that will take on a True or False 
value.

After this, is where you define all your actions. For each action,
you have your parameters with type, preconditions, and effects as
we discussed. In this case, instead of explicitly saying what states are added and deleted,
this planner lets you say what is and is not True.

## Semantics!
All actions need some kind of meaning. For the problems given to you, we will give you
the semantics as comments in the `domain.pddl` file. Make sure you understand your own. In this example, we define a rotate-i, meaning rotate a block L that is at position-i by 90 degrees counter-clockwise.

In the `domain.pddl` file, we are defining actions and effects to rotate
our "L" block.

Run:
```
pyperplan -H hff -s gbf knowledge_base/domain.pddl problems/task01.pddl
```
You shoud see some output and a `.soln` file appear next your `problems/task01.pddl` file.
The `.soln` file is the solution the planning algorithm found that takes us from our init state
to our goal state.

In general the command to run the planner has the form:

```
pyperplan -H hff -s gbf <DOMAIN path> <PROBLEM path>
```

Feel free to explore the parameters. Without the params above, the default
will be a basic BFS approach (no heuristics), but we are using heuristics with
the above.

# Step 2
For each task pddl file in problems directory, create a domain pddl or
add actions to an existing one that solve the task pddl file. Run the planner
and make sure to commit the soln file. That is, keep the soln file created by
your planner that sits next to the problem pddl file.

# Step 3
You will be given a json file that represents the initial state of the input grid.
For example, for initial state 'position-1', you will be given
```
{"input":
   [[0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0], 
    [0, 1, 1, 1]
   ]
}
```
located in the `initial_state_json` directory and in a file corresponding to the problem.
So, for example, `task01.json` has the intial input json for the problem described in `task01.pddl`.

Now, for each action you create in your `domain.pddl` file, write a Python function that actually performs the action. For example,
for a rotate-i action write a Python function like 

```python
def rotate(i, block):
# your code here
```

We will use these functions to actually execute each step in the solution provided from
the planner.

# Step 4
Write a function that takes your solution from your planner from Step 2 and the corresponding 
intial state json file in `initial_state_json` and
executes it from the functions you created in Step 3. The output
of the function should be a grid of cells with colors corresponding
to the goal state of the plan.

See the python file in `executor/executor.py` for more details and for
where you need to create your executor.

For example, if our initial state is a block L in position-1 and we have a goal to get block L in position-2, then our planner will give us the solution



```
(rotate-2 l)
```

Take (1) outputted solution and (2) the corresponding json 
input for the initial state which looks like:

```
{"input":
   [[0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 1, 0, 0], 
    [0, 1, 1, 1]
   ]
}
```
and write the execute function that take each step in the plan and transforms the
json. Our execute(plan, initial_state_json) function will 
take in that plan and initial state json and return an output json
that looks like:
```
{"output":
   [[0, 0, 0, 0],
    [1, 1, 1, 0],
    [1, 0, 0, 0], 
    [1, 0, 0, 0]
   ]
}
```

# Grading
We will take the solution file for your plan, the corresponding input json file, and 
see if the output json file produced is what is expected. See `executor/task03.json`

The points are as usual and out of 3pts total.

3 Points
1 points for code
1 point for README being accurate (see bottom)
1 point for being functionally correct

## Bonus
+3 for solving task 03 by combining task 01 and task 02. For this you need to write
the entire domain pddl and task pddl. Look at the json input and desired output.

# --> Your README instructions here

# PDDL Executor

## Overview

This script loads a PDDL domain and problem, finds a solution using breadth-first search, and applies the planned actions to update a grid until it matches the goal state.

## Files
	•	domain.pddl – Defines available actions and rules
	•	task.pddl – Specifies the initial and goal states
	•	task.json – JSON file with the initial grid state
	•	desired_output.json – JSON file with the expected goal state
	•	executor.py – Loads the PDDL plan, runs the planner, and executes actions

## Actions
	•	rotate2(grid) – Rotates the grid 90° clockwise
	•	create1(grid) – Adds a block at a specified position
	•	delete1(grid) – Removes a block

## How to Run

Execute the script with:

python executor/executor.py

The script will process multiple tasks and print whether the goal is reached.