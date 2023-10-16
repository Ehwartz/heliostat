import torch
import variables
import etas


def main():
    xys = variables.get_xys()
    Np = xys.size(0)
    zs = torch.ones(size=[Np, 1]) * 4
    xyzs = variables.get_xyzs(xys, zs)
    planes = torch.ones(size=[Np, 2]) * 6.0
    As = (planes[:, 0] * planes[:, 1]).view([Np])
    solar_collector = torch.tensor([0, 0, 84])
    dists = torch.sqrt(torch.sum(torch.square(xyzs - solar_collector), dim=-1))
    xydh = solar_collector - xyzs
    STs = torch.tensor([9.0, 10.5, 12.01, 13.5, 15.0])
    months = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) * 1.0
    Ds = torch.tensor([-59, -28, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]) * 1.0
    phi = torch.tensor([39.4]) * torch.pi * 2 / 360
    days = torch.tensor([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    E_field_y = torch.empty(size=[12])

    etas_sb_l = []
    etas_cos_l = []
    etas_trunc_l = []
    etas_optical_l = []

    for i_m in range(12):
        month = int(months[i_m])
        D = Ds[i_m]
        delta = variables.get_delta(D)
        E_field_m = torch.empty(size=[5])
        for i_ST, ST in enumerate(STs):
            omega = variables.get_omega(ST)
            alpha_s = variables.get_alpha_s(delta, phi, omega)
            delta = variables.get_delta(D)
            gamma_s = variables.get_gamma_s(alpha_s, delta, phi)
            DNI = variables.get_DNI(torch.tensor([3]), alpha_s)
            print('DNI:   ', DNI)
            etas_sb = etas.get_eta_sb(month)
            etas_cos, thetas = etas.get_eta_cos(alpha_s, gamma_s, xydh)
            etas_at = etas.get_eta_at(dists)
            etas_trunc = etas.get_eta_trunc(planes, thetas, dists, 101, 101)
            etas_ref = etas.get_eta_ref()
            eta_optical = etas_sb * etas_cos * etas_at * etas_trunc * etas_ref
            E_field = variables.get_E_field(DNI, As, eta_optical)
            E_field_m[i_ST] = E_field

            etas_sb_l.append(etas_sb)
            etas_cos_l.append(etas_cos)
            etas_trunc_l.append(etas_trunc)
            etas_optical_l.append(eta_optical)

        print(E_field_m)
        E_field_m_mean = torch.mean(E_field_m)
        E_field_y[i_m] = E_field_m_mean
    print(E_field_y)

    # torch.save(E_field_y, './E_field_y.pth')
    # print(etas_sb_l)
    # print(etas_cos_l)
    # print(etas_trunc_l)

    etas_sb_t = torch.tensor(etas_sb_l)
    etas_cos_t = torch.concat(etas_cos_l, dim=0)
    etas_trunc_t = torch.concat(etas_trunc_l, dim=0)
    etas_optical_t = torch.concat(etas_optical_l, dim=0)
    print(etas_cos_l)
    torch.save(etas_sb_t, 'etas_sb_t.pth')
    torch.save(etas_cos_t, 'etas_cos_t.pth')
    torch.save(etas_trunc_t, 'etas_trunc_t.pth')
    torch.save(etas_optical_t, 'etas_optical_t.pth')


if __name__ == '__main__':
    # main()
    days = torch.tensor([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    E_f_y = torch.load('./E_field_y.pth')
    print('E, y:   \n', E_f_y)
    E_sum = torch.sum(days * E_f_y * 5)
    print(E_f_y)
    print(E_sum)
    etas_sb_t = torch.load('etas_sb_t.pth').view([12, 5])
    etas_cos_t = torch.load('etas_cos_t.pth').view([12, 5, 1745])
    etas_trunc_t = torch.load('etas_trunc_t.pth').view([12, 5, 1745])
    etas_optical_t = torch.load('etas_optical_t.pth').view([12, 5, 1745])
    print(etas_sb_t.size())
    print(etas_cos_t.size())
    print(etas_trunc_t.size())
    print(etas_optical_t.size())
    print('sb\n', torch.mean(etas_sb_t, dim=-1))
    print('cos\n', torch.mean(torch.mean(etas_cos_t, dim=-1), dim=-1))
    print('trunc\n', torch.mean(torch.mean(etas_trunc_t, dim=-1), dim=-1))
    print('optical\n', torch.mean(torch.mean(etas_optical_t, dim=-1), dim=-1))

    print('E /m^2\n', E_f_y / (1745 * 36))

    print('sb\n', torch.mean(etas_sb_t))
    print('cos\n', torch.mean(etas_cos_t))
    print('trunc\n', torch.mean(etas_trunc_t))
    print('optical\n', torch.mean(etas_optical_t))

    print('E /m^2\n', torch.mean(E_f_y / (1745 * 36)))

    print('E, y, m^2: \n', E_f_y/(1745 * 36))

    print('E, y, y, m^2: \n', torch.mean(E_f_y)/(1745 * 36))
