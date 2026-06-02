clear

global mask 

year = 2016;     
month = 6;
day = 1;

dt0=3600*24;    % time-step (one day)

kw = 0.2;        % Extinction coefficient water (1/m)
kp = 0.02;       % Extinction coefficient algae (1/m/mg Chla)
nlgr = 1;        % nutrient-limited growth rate (dimensionless ?)

[mask,x,y,z,dx,dy,h] = getgrid;    % from ocn2
[im,jm] = size(h);
kb=length(z);
mask(find(mask~=0))=1;

z(1)=0;
z(2)=2;
z(3)=6;

% read average monthly concentration of chlorophyll-a at the ocean surface:
chl0 = read_chl(year,month);   

% read average daily light intensity at the ocean surface:
[T,I0]=read_daily(year,month,day);

chl = zeros(im,jm,kb);     % 3D chl concentration (mg/m3)
Iz = zeros(im,jm,kb);      % 3D light intensity (μE/m2/s)
Aw = zeros(im,jm,kb);      % 3D ambient algue concentration in water (no/m3)
for i=1:im
    for j=1:jm
        [Cb,s,Cmax,zmax,delz,chla_zbase,zbase]=chl_const(chl0(i,j),h(i,j),month);
        k = 1;
        while z(k)< h(i,j)
            if chl0(i,j)>0 && z(k)<=zbase
                chl(i,j,k) = (Cb - s*z(k) + Cmax*exp(-((z(k)-zmax)/delz)^2)) * chla_zbase;
                if chl(i,j,k)<0 
                    chl(i,j,k) = 0; 
                end
            end
            Ik = kw + kp * chl(i,j,k);
            Iz(i,j,k) = dt0 * I0(i,j) * exp(-Ik * z(k));
            C1 = 0.003 + 1.0154 * exp(0.05*T(i,j,k)) * exp(-0.059*Iz(i,j,k)*10^(-6)) * nlgr;  % carbon concentration
            Aw(i,j,k) = chl(i,j,k) / C1 / (2726 * 10^(-9));
            k = k + 1;
        end
    end
end

% % % plot 2D at the selected depth:
chl_plot = Aw(:,:,1);
chl_plot(mask == 0) = NaN;
figure; contourf(chl_plot',100);

% % % plot 1D at the selected point:
for_plot(:) = Aw(191,218,1:19);
figure;plot(for_plot,-z(1:19));

%--------------------------------------------------------------------------

function chl = read_chl(year,month)
global mask

dir='C:\Data\Chl_DATA\';
filename = [dir,int2str(year),'\chl',num2str(year,'%02d'),'-',num2str(month,'%02d')];
f = fopen(filename,'rb');
chl=fread(f,[310,418],'real*8');
fclose(f);
chl(mask==0) = NaN;
end

%--------------------------------------------------------------------------

function [T,sw] = read_daily(year,month,day)

dir1 = 'C:\Data\wind_data1';
dir='D:\DAY_temp_1\ocn';
    
f0=netcdf.open([dir1,'\day',int2str(year),'-',num2str(month,'%02d'),'-',num2str(day,'%02d'),'.nc'],'NOWRITE');
var_sw=netcdf.inqVarID(f0,'light_int');
sw=double(netcdf.getVar(f0,var_sw)); sw(find(sw>1e+10))=0;
netcdf.close(f0);

f0=netcdf.open([dir,'\day',int2str(year),'-',num2str(month,'%02d'),'-',num2str(day,'%02d'),'.nc'],'NOWRITE');
var_T=netcdf.inqVarID(f0,'temp');
T=double(netcdf.getVar(f0,var_T)); T(find(T>1e+10))=NaN;
netcdf.close(f0);

end

%--------------------------------------------------------------------------

function [Cb,s,Cmax,zmax,delz,chla_zbase,zbase]=chl_const(chl,depth,month)
% Constants for vertical chl profile according to [Ardyna,2013]
% Selection of constants depends on:
% - model depth at the point 
% - surface chl concentration
% - season (month)

if chl > 0
    if depth > 50
        if chl<0.1
            if month >=2 && month <=4
                Cb = 0.8356;
                s = 0.0026;
                Cmax = 0.945;
                zmax = 3.83;
                delz = 22.21;
                chla_zbase = 0.0285;
            elseif month >=5 && month <=9
                Cb = 0.4908;
                s = 0.0019;
                Cmax = 1.2039;
                zmax = 48.07;
                delz = 26.43;
                chla_zbase = 0.0935;
            else
                Cb = 1.1696;
                s = 0.0045;
                Cmax = 0.113;
                zmax = 83.42;
                delz = 24.99;
                chla_zbase = 0.0427;
            end
            zbase = 110;
        elseif chl<0.3
            if month >=2 && month <=4
                Cb = 0.7272;
                s = 0.0009;
                Cmax = 0.8371;
                zmax = 0;
                delz = 36.2;
                chla_zbase = 0.0959;
            elseif month >=5 && month <=9
                Cb = 0.6087;
                s = 0.0026;
                Cmax = 0.9656;
                zmax = 36.05;
                delz = 27.27;
                chla_zbase = 0.1931;
            else
                Cb = 0.6519;
                s = 0.003;
                Cmax = 0.7873;
                zmax = 2.37;
                delz = 63.03;
                chla_zbase = 0.1043;
            end
            zbase = 80;
        elseif chl<0.5
            if month >=2 && month <=4
                Cb = 0.4542;
                s = 0.0007;
                Cmax = 0.8127;
                zmax = 1.91;
                delz = 80.52;
                chla_zbase = 0.2764;
            elseif month >=5 && month <=9
                Cb = 0.5461;
                s = 0.0016;
                Cmax = 1.0198;
                zmax = 23.81;
                delz = 28.47;
                chla_zbase = 0.3324;
            else
                Cb = 0.0939;
                s = 0.0001;
                Cmax = 1.4592;
                zmax = 1.34;
                delz = 66.32;
                chla_zbase = 0.2254;
            end
            zbase = 60;
        elseif chl<0.7
            if month >=2 && month <=4
                Cb = 0.4751;
                s = 0.0013;
                Cmax = 0.9337;
                zmax = 0;
                delz = 68.35;
                chla_zbase = 0.3904;
            elseif month >=5 && month <=9
                Cb = 0.5093;
                s = 0.0017;
                Cmax = 1.1552;
                zmax = 17.77;
                delz = 30.12;
                chla_zbase = 0.4151;
            else
                Cb = 0.3126;
                s = 0.0013;
                Cmax = 1.3075;
                zmax = 0;
                delz = 54.03;
                chla_zbase = 0.3395;
            end
            zbase = 55;
        elseif chl<1
            Cb = 0.5449;
            s = 0.0023;
            Cmax = 1.1564;
            zmax = 15.68;
            delz = 31.69;
            zbase = 50;
            chla_zbase = 0.5172;
        elseif chl<3
            Cb = 0.4611;
            s = 0.002;
            Cmax = 1.4783;
            zmax = 4.81;
            delz = 35.92;
            zbase = 35;
            chla_zbase = 0.7841;
        elseif chl<8
            Cb = 0.487;
            s = 0.0024;
            Cmax = 1.7256;
            zmax = 0;
            delz = 31.76;
            zbase = 25;
            chla_zbase = 1.8078;
        else
            Cb = 0.3987;
            s = 0.0019;
            Cmax = 2.1463;
            zmax = 6.64;
            delz = 18.45;
            zbase = 10;
            chla_zbase = 4.3778;
        end
    else
        if chl<0.1
            if month >=2 && month <=4
                Cb = 0.9949;
                s = 0.0113;
                Cmax = 0.255;
                zmax = 0.9621;
                delz = 0.1014;
                chla_zbase = 0.0503;
            elseif month >=5 && month <=9
                Cb = 0.0001;
                s = 1.6112;
                Cmax = 4.4054;
                zmax = 1.1616;
                delz = 0.6773;
                chla_zbase = 0.2149;
            else
                Cb = 0.9965;
                s = 0.5444;
                Cmax = 0.7487;
                zmax = 0.8438;
                delz = 0.2959;
                chla_zbase = 0.0502;
            end
            zbase = 50;
        elseif chl<0.3
            if month >=2 && month <=4
                Cb = 0.9949;
                s = 0.0113;
                Cmax = 0.255;
                zmax = 0.9621;
                delz = 0.1014;
                chla_zbase = 0.1508;
            elseif month >=5 && month <=9
                Cb = 0.0001;
                s = 2.8568;
                Cmax = 4.4586;
                zmax = 1.0266;
                delz = 0.6895;
                chla_zbase = 0.3087;
            else
                Cb = 0.9965;
                s = 0.5444;
                Cmax = 0.7487;
                zmax = 0.8438;
                delz = 0.2959;
                chla_zbase = 0.1505;
            end
            zbase = 50;
        elseif chl<0.5
            Cb = 0.0001;
            s = 2.4886;
            Cmax = 3.8592;
            zmax = 1.0916;
            delz = 0.8220;
            zbase = 50;
            chla_zbase = 0.5289;
        elseif chl<0.7
            Cb = 0.715;
            s = 0;
            Cmax = 0.8592;
            zmax = 1.3961;
            delz = 0.8428;
            zbase = 50;
            chla_zbase = 0.714;
        elseif chl<1
            Cb = 0.7990;
            s = 0;
            Cmax = 0.3761;
            zmax = 0.7589;
            delz = 0.4448;
            zbase = 50;
            chla_zbase = 0.9152;
        elseif chl<3
            Cb = 0.0001;
            s = 1.4083;
            Cmax = 2.1591;
            zmax = 1.1605;
            delz = 1.4467;
            zbase = 35;
            chla_zbase = 1.322;
        elseif chl<8
            Cb = 1.0555;
            s = 0.3629;
            Cmax = 0.2359;
            zmax = 0.2402;
            delz = 0.2483;
            zbase = 25;
            chla_zbase = 3.4842;
        else
            Cb = 1.0196;
            s = 0.7762;
            Cmax = 0.866;
            zmax = 0.2144;
            delz = 0.1637;
            zbase = 10;
            chla_zbase = 8.5078;
        end
    end
    zbase = 150;
else
    Cb = 0;
    s = 0;
    Cmax = 0;
    zmax = 0;
    delz = 1;
    zbase = 0;
    chla_zbase = 0;
end
end