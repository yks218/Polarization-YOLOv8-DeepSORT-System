function dop
% 设置入射出射相关角
al1 = cfg.al1;   %光源入射坐标到坐标原点的俯仰角，范围[-pi/2,pi/2]
be1 = cfg.be1;   %光源入射坐标到坐标原点的水平角（方向角），范围[0,2*pi]
al2= cfg.al2;   %探测器像面中心坐标到坐标原点的俯仰角，范围[-pi/2,pi/2]
be2= cfg.al2;   %探测器像面中心坐标到坐标原点的水平角（方向角），范围[0,2*pi]
e=1;
d=1;
% 设置材质属性

data = readmatrix(cfg.excel_path);
index = 1;
numRows = size(data, 1);
intensitys = data(:, 5);

for i = 1:numRows
    wavelength = data(i, 1);
    n2 = data(i, 2);
    k2 = data(i, 3);
    camera  = data(i, 4);
intensity= data(i, 5);
PT = data(i, 6);
MP0=0.5*[PT,PT,0,0;PT,PT,0,0;0,0,PT,0;0,0,0,0];
MP45=0.5*[PT,0,PT,0;0,PT,0,0;PT,0,PT,0;0,0,0,0];
MP90=0.5*[PT,-PT,0,0;-PT,PT,0,0;0,0,PT,0;0,0,0,0];
MP135=0.5*[PT,0,-PT,0;0,PT,0,0;-PT,0,PT,0;0,0,0,0];

roughness = cfg.roughness; % 粗糙度

P_LED = cfg.P_LED; % LED 电功率 3W

theta = cfg.theta; % 发散角 120度

% 计算 LED 发射光强度
A_emit = pi * (1^2) * (theta / 360); 

% 计算总的光强度
I_total = P_LED / A_emit; 

total_intensity = sum(intensitys); % 总的光强度
S00 = I_total * (intensity / total_intensity); % 每个波长下的入射光强度

n1=1;         %大气折射率

positionTarget=[0, 0, 0];              %目标的基准坐标
positionLaser=[e*cosd(al1)*cosd(be1),e*cosd(al1)*sind(be1),e*sind(al1)]; %入射光源的空间坐标
positionDetector=[d*cosd(al2)*cosd(be2),e*cosd(al2)*sind(be2),e*sind(al2)]; %探测器像面中心的三维坐标  单位：毫米

Laservector=positionTarget-positionLaser; %入射光向量
LasV=Laservector/norm(Laservector);       %单位化

Detvector=positionDetector-positionTarget; %探测器观测向量
DetV=Detvector/norm(Detvector);       %单位化

azimuth_incident = atan2(-LasV(2), -LasV(1)); % 入射光的周向角
azimuth_reflection = atan2(DetV(2), DetV(1)); % 反射光的周向角
delta_azimuth_rad = mod(azimuth_reflection - azimuth_incident, 2*pi); % 保证在[0, 2π]范围内
delta_azimuth_deg = rad2deg(delta_azimuth_rad); % 转换为度数

v=[0 0 1];%面法向量
    thetsky=acosd(dot(v,-LasV)/norm(v));  %以该网格法向量为轴的入射光天顶角
    thetg=acosd(dot(v,DetV)/norm(v));    %以该网格法向量为轴的观测方向天顶角

%Mueller矩阵
B2=cosd(thetsky)*cosd(thetg)+sind(thetsky)*sind(thetg)*cosd(delta_azimuth_deg);%入射角反射角夹角
B=(acosd(B2))/2;
thetaN=acosd((cosd(thetsky)+cosd(thetg))/(2*cosd(B)));%微元法线与宏观法线夹角
%琼斯尔缪勒矩阵

etai1=(1-cos(thetsky)*cos(B))/(sin(thetsky)*sin(B));
etai=acosd(etai1);
etar2=(1-cos(thetg)*cos(B))/(sin(thetg)*sin(B));
etar=acosd(etar2);
D=n2^2-k2^2-sind(B)^2;
C=4*n2^2*k2^2+D^2;
B22=sqrt((sqrt(C)-D)/2);
A=sqrt((sqrt(C)+D)/2);
rs=sqrt(((A-cosd(B))^2+B22^2)/((A+cosd(B))^2+B22^2));
rp=rs*sqrt(((A-sind(B))^2+B22^2)/((A+sind(B))^2+B22^2));
Rs=rs^2;
Rp=rp^2;
R=(Rs+Rp)/2;
T=[cosd(etar),cosd(etai);-sind(etar),cosd(etar)]*[rs,0;0,rp]*[cosd(etai),sind(etai);-sind(etai),cosd(etai)];
Tss=T(1,1);
Tps=T(1,2);
Tsp=T(2,1);
Tpp=T(2,2);
%菲涅尔缪勒矩阵
m00=((norm(Tss))^2+(norm(Tsp))^2+(norm(Tps))^2+(norm(Tpp))^2)/2;
m01=((norm(Tss))^2+(norm(Tsp))^2-(norm(Tps))^2-(norm(Tpp))^2)/2;
m02=(Tss*conj(Tps)+conj(Tss)*Tps+Tsp*conj(Tpp)+conj(Tsp)*Tpp)/2;
m03=(1i*(Tps*conj(Tss)+conj(Tss)*Tps)+1i*(Tpp*conj(Tsp)+conj(Tsp)*Tpp))/2;
m10=((norm(Tss))^2-(norm(Tsp))^2+(norm(Tps))^2-(norm(Tpp))^2)/2;
m11=((norm(Tss))^2-(norm(Tsp))^2-(norm(Tps))^2+(norm(Tpp))^2)/2;
m12=(Tss*conj(Tps)+conj(Tss)*Tps-Tsp*conj(Tpp)-conj(Tsp)*Tpp)/2;
m13=(1i*(Tps*conj(Tss)+conj(Tss)*Tps)-1i*(Tpp*conj(Tsp)+conj(Tsp)*Tpp))/2;
m20=(Tss*conj(Tsp)+conj(Tss)*Tsp+Tps*conj(Tpp)+conj(Tps)*Tpp)/2;
m21=(Tss*conj(Tsp)+conj(Tss)*Tsp-Tps*conj(Tpp)-conj(Tps)*Tpp)/2;
m22=(Tss*conj(Tpp)+conj(Tss)*Tps+Tps*conj(Tsp)+conj(Tsp)*Tpp)/2;
m23=(1i*(Tps*conj(Tsp)-conj(Tss)*Tps)-1i*(Tss*conj(Tpp)-conj(Tsp)*Tpp))/2;
m30=(1i*(Tss*conj(Tsp)-conj(Tss)*Tps)+1i*(Tps*conj(Tpp)-conj(Tsp)*Tpp))/2;
m31=(1i*(Tss*conj(Tsp)-conj(Tss)*Tps)-1i*(Tps*conj(Tpp)-conj(Tsp)*Tpp))/2;
m32=(1i*(Tss*conj(Tpp)-conj(Tss)*Tps)+1i*(Tps*conj(Tsp)-conj(Tsp)*Tpp))/2;
m33=((Tss*conj(Tpp)+conj(Tss)*Tps)-(Tps*conj(Tsp)+conj(Tsp)*Tpp))/2;
M=[m00,m01,m02,m03;m10,m11,m12,m13;m20,m21,m22,m23;m30,m31,m32,m33];
p=1/(2*pi*(roughness.^2)*((cosd(thetaN)).^3))*exp(-(tand(thetaN))^2/(2*roughness));%微元概率密度函数

 NdotL = max(dot(v, -LasV), 0);
 NdotV = max(dot(v, DetV), 0);
  G1_V = 2 .* NdotV ./ (NdotV + sqrt(roughness.^2 + (1 - roughness.^2) .* (NdotV.^2)));
    G1_L = 2 .* NdotL ./ (NdotL + sqrt(roughness.^2 + (1 - roughness.^2) .* (NdotL.^2)));
    G = G1_V .* G1_L;

ks=R*exp(-(4*pi*roughness*cosd(thetsky))^2/(( wavelength*10^(-2))^2));
pbrdf=R*G*p/(4*cosd(thetaN)*cosd(thetsky)*cosd(thetg))*M;

F=R+(1-R)*(1-cosd(thetsky))^5;
fs=F*G*p/(4*cosd(thetsky)*cosd(thetg));

rho=1-R;
pBRDF_Lambertian = rho/pi;

alpha = max(thetsky, thetg);
beta = min(thetsky, thetg);
sigma2 = roughness^2;

AA = 1 - (0.5 * sigma2) / (sigma2 + 0.33);
BB = 0.45 * sigma2 / (sigma2 + 0.09);
cos_phi_d = cos(0);
sin_alpha = sin(alpha);
tan_beta = tan(beta);
oren_nayar_correction = AA + BB * max(0, cos_phi_d * sin_alpha * tan_beta);
pBRDF11 = pBRDF_Lambertian + oren_nayar_correction;
pbrdf=pbrdf*S00*camera;
T=298;
Ie=3.7416*10^8*wavelength^(-5)/((exp(1.4388*10^4/(wavelength*T))-1)*pi);

epsilon=[1-pbrdf(1,1);-pbrdf(2,1);-pbrdf(3,1);-pbrdf(4,1)].*Ie;
S=[pbrdf(1,1);pbrdf(2,1);pbrdf(3,1);pbrdf(4,1)];
S0=MP0*S;
S45=MP45*S;
S90=MP90*S;
S135=MP135*S;
S(1)=S0(1)+S90(1);
S(2)=S0(1)-S90(1);
S(3)=S45(1)-S135(1);
S(4)=0;
S(1)=S(1)+pBRDF11;
S(1)=S(1)+fs;

dop(index)=norm(sqrt(S (2)^2+S (3)^2+S (4)^2)/S (1));

index = index + 1; 

end

end
