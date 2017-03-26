aoa = linspace(5,50,50);
cl = zeros(1,50);
cl(11:50) = 0.025*aoa(11:50);
m = cl(11)/(aoa(11)-aoa(1));
b = -m*aoa(1);
cl(1:10) = m*aoa(1:10)+b;

cd = .001*(aoa).^2;
clcd = cl./cd;

plot(aoa,cl,'g',aoa,cd,'r',aoa,clcd,'k')