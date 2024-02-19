# --------------------------------------------------
# 文件名: gurobiTest
# 创建时间: 2024/2/18 23:29
# 描述:
# 作者: WangYuanbo
# --------------------------------------------------
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
# Create a new model
m = gp.Model("mip1")
# Create variables
x = m.addVar(vtype=GRB.BINARY, name="x")
y = m.addVar(vtype=GRB.BINARY, name="y")
z = m.addVar(vtype=GRB.BINARY, name="z")
# Create variables
x = m.addMVar(shape=3, vtype=GRB.BINARY, name="x")
# Add constraint: x + 2 y + 3 z <= 4
m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
# Add constraint: x + y >= 1
m.addConstr(x + y >= 1, "c1")
# Optimize model
m.optimize()
for v in m.getVars():
    print('%s %g' % (v.VarName, v.X))
print('Obj: %g' % m.ObjVal)