import json
from pyperplan.pddl.parser import Parser
from pyperplan.task import Task
from pyperplan.search import breadth_first_search
from pyperplan.grounding import ground
from pyperplan.heuristics.blind import BlindHeuristic

def load_pddl(domain_path, problem_path):
   parser = Parser(domain_path, problem_path)
   domain = parser.parse_domain()
   problem = parser.parse_problem(domain)

   return domain, problem

# rotates 90 clockwise
def rotate2(grid):
   new = [list(row) for row in zip(*grid[::-1])]
   return new

def create1(grid):
   new = grid.copy()
   new[1][3] = 1
   return new

def delete1(grid):
   new = grid.copy()
   new[1][3] = 0
   return new

# def execute(path_to_goal_state, domain_path, problem_path, initial_state_json):
def execute(domain_path, problem_path, initial_state_json, desired_output_json):

   """
   Given a plan produced from your planner, execute that plan where each
   action corresponds to a python function or a parameterization of that function

   :param plan: plan to soln file that has the actions produced from planner
   :param initial_state_json: initial state json file
   :return:
      a grid of cells with color (the predicted output with colors ranging from 0-9) in json format as shown below in example:

         {"output":
            [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]}
   """
   with open(initial_state_json, 'r') as f:
      initial = json.load(f)

   with open(desired_output_json, 'r') as f:
      final = json.load(f)
   
   grid = initial['input']
   goal = final['output']
   
   domain, problem = load_pddl(domain_path, problem_path)

   # Ground into a Task
   task = ground(problem)
   print(task)

   plan = breadth_first_search(task)
   print("Plan: ", plan)
   plan_clean = [op.name for op in plan]
   print(plan_clean)

   steps = 0
   for action in plan:
      if action.name in ['(rotate-2 l)', '(rotate-5 b)']:
         new = rotate2(grid)
      if action.name == '(create-1 b)':
         new = create1(grid)
      if action.name == '(delete-1 b)':
         new = delete1(grid)
      steps += 1
   
   if new == goal:
      print(f"Goal reached in {steps} steps!")      

print("Task 01: \n")
execute("knowledge_base/domain.pddl", "problems/task01.pddl", "initial_state_json/task01.json", "executor/desired_output/task01.json")
print("Task 02: \n")
execute("knowledge_base/domain_B.pddl", "problems/task02.pddl", "initial_state_json/task02.json", "executor/desired_output/task02.json")
print("Task 03: \n")
execute("knowledge_base/domain_final.pddl", "problems/task03.pddl", "initial_state_json/task03.json", "executor/desired_output/task03.json")