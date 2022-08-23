import numpy as np
import re
from simplex import Model


def bfs(g):
    queue = []
    visited = set([0])
    edges = np.argwhere(g[0] < 0)[:, 0]
    g[0] = 0
    queue.extend(edges)
    while queue:
        e = queue.pop(0)
        e_dst = np.argwhere(g[:, e] > 0)
        if e_dst.size > 0:
            visited.add(e_dst[0][0])
            queue.extend(np.argwhere(g[e_dst[0][0]] < 0)[:, 0])
            g[e_dst[0][0]] = 0
    return np.array(list(visited))


def to_dot(g, weights):
    dot = "digraph G {\n"
    for i, col in enumerate(g.T):
        src = np.argwhere(col < 0)
        dst = np.argwhere(col > 0)
        if src.size and dst.size:
            src = src[0][0] + 1
            dst = dst[0][0] + 1
            dot += f"{src}->{dst} [label=\"{weights[i]}\"];\n"
    dot += "}"
    with open("graph.dot", "w") as f:
        f.write(dot)


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore')
    np.set_printoptions(linewidth=np.inf)

    nvertices, nedges = np.array(re.findall(r"\d+", input())).astype(np.uint8)
    caps = np.array(re.findall(r"-?\d+", input())).astype(np.longdouble)

    m = np.zeros((nvertices, nedges), np.longdouble)
    for i in range(nvertices):
        m[i] = re.findall(r"-?\d+", input())
    c = -m[0]
    n = m[1:-1]

    model = Model()
    model.set_objective(c)
    for row in n:
        model.add_constraint(row, 0, True)

    cap_cons = np.identity(nedges, np.longdouble)
    for row, cap in zip(cap_cons, caps):
        model.add_constraint(row, cap)

    sol = model.solve()
    res_cap = caps - sol[2]
    res_m = -m * sol[2]
    m = m * res_cap
    residual = np.hstack((m, res_m))
    min_cut_v = bfs(residual)
    min_cut = np.zeros((nvertices,))
    min_cut[min_cut_v] = 1
    print(int(sol[1]))
    out = ""
    for n in sol[2]:
        out += f"{n:.0f} "
    print(out)
    out = ""
    for n in min_cut:
        out += f"{n:.0f} "
    print(out)
