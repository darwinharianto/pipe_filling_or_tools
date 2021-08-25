from ortools.sat.python import cp_model


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


model = cp_model.CpModel()

all_s = 20
select_s = 5
# bias = 3
print("bias","\t", "x","\t", "check_ones","\t", "ans")
for bias in range(20):

    lower = bias
    upper = min(select_s+bias, all_s)
    s = {}
    for i in range(all_s):
        s[i] = model.NewBoolVar(str(i))

    check_ones = model.NewBoolVar("check_ones")
    # y = model.NewIntVar(0, all_s, "y")
    # z = model.NewIntVar(0, 10, "z")

    model.Add(sum(s[i] for i in range(20)) <= 20)
    model.Add(sum(s[i] for i in range(20)) >= 6)
    model.Add(sum(s[i] for i in range(3)) == 1)
    model.Add(sum(s[i] for i in range(6)) == 1)
    model.Add(sum(s[i] for i in range(8)) == 3)
    sum_zero = 0
    sum_one = 0

    x = model.NewIntVar(0, all_s, "x")  # 4x1
    v1 = [model.NewIntVar(0, all_s, "") for _ in range(all_s)]  # 10x4
    for i in range(0, all_s):
        # i = j - bias
        model.Add(v1[i] == all_s + s[clamp(bias+i + 1, 0,  all_s-1)]*(i - all_s))
        # model.Add(v1[i] == select_s-s[i+bias]*(select_s-i))
        # model.Add(v1[i] == s[i+bias]*(i))
    model.Add(sum(s[i] for i in range(bias + 1, clamp(
        bias+select_s + 1, 0,  all_s))) > 0).OnlyEnforceIf(check_ones)
    model.Add(sum(s[i] for i in range(bias + 1, clamp(
        bias+select_s + 1, 0,  all_s))) == 0).OnlyEnforceIf(check_ones.Not())
    model.AddMinEquality(x, v1)
    # .OnlyEnforceIf(check_ones)
    # model.AddMaxEquality(x, v1)
    solver = cp_model.CpSolver()
    solver.Solve(model)

    ans = []
    for i in range(all_s):
        ans.append(solver.Value(s[i]))
    print(bias,"\t", solver.Value(x)+1,"\t", solver.Value(check_ones),"\t", ans, "\t", (solver.Value(x)+1 if  solver.Value(check_ones) else "-"))