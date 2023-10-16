import torch
import variables

def gaussian2(xy, sigma):
    """

    :param xy: points, shape [Np, nw*nh, 2]
    :param sigma: shape [Np, 1]
    :return:
    """
    return torch.exp(-torch.sum(torch.square(xy), dim=-1) / (2 * torch.square(sigma)))


def get_eta_sb(month, etas_sb=variables.etas_sb):
    # etas_sb = {1: 1 - 0.0250 / 2,
    #            2: 1 - 0.0157 / 2,
    #            3: 1 - 0.0152 / 2,
    #            4: 1 - 0.0167 / 2,
    #            5: 1 - 0.0179 / 2,
    #            6: 1 - 0.0179 / 2,
    #            7: 1 - 0.0179 / 2,
    #            8: 1 - 0.0168 / 2,
    #            9: 1 - 0.0152 / 2,
    #            10: 1 - 0.0158 / 2,
    #            11: 1 - 0.0253 / 2,
    #            12: 1 - 0.0364 / 2}
    return etas_sb.get(month)


def get_eta_cos(alpha_s, gamma_s, xydh):
    Np = xydh.size(0)
    Iv = torch.tensor([torch.cos(alpha_s) * torch.sin(gamma_s),
                       torch.cos(alpha_s) * torch.cos(gamma_s),
                       -torch.sin(alpha_s)])

    # tmp = torch.zeros(size=[n, 3])
    # tmp[:, 2] = 84
    # xydh = tmp - xyz
    Rv = xydh / torch.sqrt(torch.sum(torch.square(xydh), dim=1).view([Np, 1]))
    theta = 0.5 * torch.arccos(torch.sum(-Iv * Rv, dim=-1))

    return torch.cos(theta), theta.view([Np])


def get_eta_at(dists):
    return 0.99321 - 0.0001176 * dists + 1.97e-8 * torch.square(dists)


def get_eta_trunc(planes, thetas, dists, nw, nh):
    Np = planes.size(0)
    ws = planes[:, 0].view([Np, 1])
    hs = planes[:, 1].view([Np, 1])

    dws = (ws / nw).view([Np, 1])
    dhs = (hs / nh).view([Np, 1])
    dss = dws * dhs
    d = torch.sqrt(ws * hs).view([Np])
    S_rec_f = 0
    H_t = d * torch.abs(S_rec_f - torch.cos(thetas))
    W_s = d * torch.abs(S_rec_f * torch.cos(thetas) - 1)
    sigma_ast = torch.sqrt(0.5 * (torch.square(H_t) + torch.square(W_s))) / (4 * dists)
    sigma_sun = torch.tensor([2.09e-3])
    sigma_t = torch.tensor([0])
    sigma_bq = torch.tensor([0])
    # shape [Np, 1]
    sigma_tot = torch.sqrt(torch.square(dists) * (torch.square(sigma_sun) +
                                                  torch.square(sigma_bq) +
                                                  torch.square(sigma_ast) +
                                                  torch.square(sigma_t))).view([Np, 1])
    xy_plane = planes2points(planes, nw, nh)
    return integrate(xy_plane, sigma_tot, dss) / (2 * torch.pi * torch.square(sigma_tot).view([Np]))


def get_eta_ref():
    return torch.tensor([0.92])


# planes: Np * [w, h]
# nw, nh:  n * 10^n + 1
def planes2points(planes, nw, nh):
    """

    :param planes:
    :param nw:
    :param nh:
    :return points, shape [Np, nw*nh, 2]
    """
    Np = planes.size(0)
    ws = planes[:, 0].view([Np, 1])
    hs = planes[:, 1].view([Np, 1])
    n_pts = nw * nh
    lin_w = torch.linspace(-0.5, 0.5, nw)
    lin_h = torch.linspace(-0.5, 0.5, nh)
    grid = torch.meshgrid([lin_w, lin_h], indexing='ij')
    xy_grid = torch.stack(grid).permute(1, 2, 0).view(-1, 2)
    xy_plane = torch.empty(size=[Np, n_pts, 2])
    xy_plane[:, :, 0] = ws * xy_grid[:, 0]
    xy_plane[:, :, 1] = hs * xy_grid[:, 1]
    return xy_plane


def integrate(xy, sigma, dss):
    """

    :param xy: points, shape [Np, nw*nh, 2]
    :param sigma: shape [Np, 1]
    :param dss: shape [Np, 1]
    :return:
    """
    gaus = gaussian2(xy, sigma)
    return torch.sum(gaus * dss, dim=-1)


if __name__ == '__main__':
    # N = 1700
    # planes = torch.ones(size=[N, 2]) * 6
    # thetas = torch.rand(size=[N, 1])
    # dists = torch.rand(size=[N, 1]) * 10
    # nw = 101
    # nh = 101
    # eta_trunc = get_eta_trunc(planes, thetas, dists, nw, nh)
    print(get_eta_sb(11))




