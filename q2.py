import torch
import variables
import etas


class OptM(torch.nn.Module):
    def __init__(self):
        super(OptM, self).__init__()
        self.lh = torch.nn.Parameter(torch.tensor([6.0]))
        self.h = torch.nn.Parameter(torch.tensor([4.0]))

    def forward(self, xy_hel, Rd):
        Np = xy_hel.size(0)
        planes = torch.zeros(size=[Np, 2])
        w = Rd - 2.5
        planes[:, 0] = planes[:, 0] + w
        planes[:, 1] = planes[:, 1] + self.lh
        As = (planes[:, 0] * planes[:, 1]).view([Np])
        xyzs = torch.zeros(size=[Np, 3])
        xyzs[:, :2] = xyzs[:, :2] + xy_hel
        xyzs[:, 2] = xyzs[:, 2] + self.h
        solar_collector = torch.tensor([0, 0, 84])
        solar_collector.requires_grad_(False)
        dists = torch.sqrt(torch.sum(torch.square(xyzs - solar_collector), dim=-1))
        xydh = solar_collector - xyzs
        STs = torch.tensor([9.0, 10.5, 12.01, 13.5, 15.0])
        months = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) * 1.0
        Ds = torch.tensor([-59, -28, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275]) * 1.0
        phi = torch.tensor([39.4]) * torch.pi * 2 / 360
        E_sum = torch.tensor([0.0])
        E_sum.requires_grad_(True)
        for i_m in range(12):
            print('month:   ', i_m)
            month = int(months[i_m])
            D = Ds[i_m]
            delta = variables.get_delta(D)

            for i_ST, ST in enumerate(STs):
                print('   ST:  ', ST)
                omega = variables.get_omega(ST)
                alpha_s = variables.get_alpha_s(delta, phi, omega)
                delta = variables.get_delta(D)
                gamma_s = variables.get_gamma_s(alpha_s, delta, phi)
                DNI = variables.get_DNI(torch.tensor([3]), alpha_s)

                etas_sb = etas.get_eta_sb(month)
                etas_cos, thetas = etas.get_eta_cos(alpha_s, gamma_s, xydh)
                etas_at = etas.get_eta_at(dists)
                etas_trunc = etas.get_eta_trunc(planes, thetas, dists, 101, 101)
                etas_ref = etas.get_eta_ref()
                eta_optical = etas_sb * etas_cos * etas_at * etas_trunc * etas_ref
                E_field = variables.get_E_field(DNI, As, eta_optical)
                E_sum = E_sum + E_field
        return - E_sum / 60


if __name__ == '__main__':
    model = OptM()
    xy_hel = torch.load('./layouts/layout299.pth')
    Rds = torch.arange(3.5, 6.5, 0.01)
    es = model(xy_hel, Rds[-1])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)
    epoch = 300
    for i in range(epoch):
        optimizer.zero_grad()
        es = model(xy_hel, Rds[-1])
        es.backward()
        optimizer.step()
        print(es)
