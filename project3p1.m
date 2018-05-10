clear all;
clc;
close all;
% Data is a matrix that is (N x M) in size, where there are N samples and M
% different dimensions
DataStruct = importdata('Two_Class_FourDGaussians.dat');
x = DataStruct.data();
y = x(:, 5);
x = x(:, 1:end-1);
[EIGVEC, EIGVAL] = eig(cov(x));
%Use PCA to reduce to 2 dimensions
EIGVEC = EIGVEC(:, end:-1:1);
EIGVAL = EIGVAL(:, end:-1:1);
EVr = EIGVEC(:,1:2);
x = x * EVr;

%plot(XT(:,1),XT(:,2),'or'), pause


w = zeros(2,1);
eta = 0.0000002;

for i=5001:10000
    if(y(i)==2)
       y(i)=-1; 
    end
end
%% Finding the LMS error to modify the weights and get the best results
for im=1:100   % this is the number of implementations
    for i=2:9999 % this is the number of iterations
        e(i)=y(i)-x(i-1,:)*w;
        w=w+eta*e(i)*x(i-1);
    end
        e100(im,:)=e;
end
semilogy(sum(e100.^2)/100,'g');
%

%% Use the finalized weights to determine the linear separation

c1=0;
c2=0;
c1true=0;
c2true=0;
c1wrong=0;
c2wrong=0;
bias = 0;
for i=2:9999
   newe(i)=y(i)-x(i-1,:)*w; 
   if(newe(i)>bias)
       c1=c1+1;
       if(y(i)==1)
          c1true=c1true+1; 
       end
       if(y(i)==-1)
          c1wrong=c1wrong+1; 
       end
   end
   if(newe(i)<bias)
      c2=c2+1;
      if(y(i)==-1)
        c2true=c2true+1;
      end
      if(y(i)==1)
         c2wrong=c2wrong+1; 
      end
   end
end

c1cr=c1true/(c1true+c2wrong);
c2cr=c2true/(c2true+c1wrong);
CR = 100*((c1cr+c2cr)/2);

LC(1)=1;
for i=2:9999
   lce(i) = y(i)- x(i-1,:)*w;
   LC(i) = LC(i-1)-lce(i);
end

