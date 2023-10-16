import torch
import matplotlib.pyplot as plt


def new_ring_region(r_start, Rd):
    n = torch.ceil(torch.pi / torch.arcsin(Rd / r_start))
    rs = []
    d_theta = torch.pi / n
    r_tmp = Rd / torch.sin(d_theta)
    # print('tmp   ', r_tmp)
    rs.append(r_tmp)
    thetas = torch.arange(int(n)) * d_theta * 2
    # print(thetas)
    xys = torch.stack([torch.cos(thetas), torch.sin(thetas)]).permute([1, 0]) * r_tmp
    # print(xys.size())
    r_next = get_next_r(r_tmp, Rd, d_theta)
    if r_tmp > 350:
        return xys, r_next
    else:
        r_tmp = r_next
    rs.append(r_tmp)
    # print('tmp   ', r_tmp)

    while not go_next_region(r_tmp, Rd, d_theta):
        #
        thetas = thetas + d_theta
        xys_new = torch.stack([torch.cos(thetas), torch.sin(thetas)]).permute([1, 0]) * r_tmp
        xys = torch.concat([xys, xys_new], dim=0)
        r_next = get_next_r(r_tmp, Rd, d_theta)
        if float(r_next) > 350:
            break
        if len(rs) > 2 and float(r_next - rs[-2]) < float(2 * Rd):
            break
        if float(r_next) > float(r_tmp):
            r_tmp = r_next
        else:
            break
        rs.append(r_tmp)
    return xys, r_tmp


def get_next_r(r, Rd, d_theta):
    return r * torch.cos(d_theta) + torch.sqrt(torch.square(2 * Rd) -
                                               torch.square(r * torch.sin(d_theta)))


def go_next_region(r, Rd, d_theta):
    return float(r * torch.sin(d_theta)) >= float(Rd) * 2


def generate_regions(Rd: torch.Tensor):
    xyss = []
    r_start = 100 + Rd
    xys, r_last = new_ring_region(r_start, Rd)

    xyss.append(xys)
    while r_last < 350:
        r_start = r_last + Rd * 2
        if r_start > 350:
            break
        xys, r_last = new_ring_region(r_start, Rd)
        xyss.append(xys)

    return torch.concat(xyss, dim=0)


if __name__ == '__main__':
    # xys, r_last = new_ring_region(torch.tensor(100), torch.tensor(6))
    # print(xys.size())
    # print(r_last)
    # plt.figure(figsize=[640, 640])
    # plt.scatter(xys[:, 0], xys[:, 1])
    # plt.show()
    # rings = generate_regions(torch.tensor(4))
    # print(rings.size())
    # plt.scatter(rings[:, 0], rings[:, 1])
    # plt.show()
    # print(torch.sqrt(torch.sum(torch.square(rings), dim=-1)))
    Rds = torch.arange(3.5, 6.5, 0.01)
    for i, Rd in enumerate(Rds):
        print(i)
        rings = generate_regions(Rd)
        print(rings.size())
        torch.save(rings, f'./layouts/layout{i}.pth')
