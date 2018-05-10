
for Kcount=0:1
for Ccount=0:2
    if(Ccount==0)
        Cval=1;
    end
    if(Ccount==1)
        Cval=10;
    end
    if(Ccount==2)
        Cval=100;
    end
I = imread('p2data.bmp');
I = I(:,:,1);
[C1x,C1y] = ind2sub(size(I),find(I==0));
[C2x,C2y] = ind2sub(size(I),find(I==255));
C1 = [C1x C1y];
C2 = [C2x C2y];
Data = [C1x C1y; C2x C2y];
Data = Data(randperm(size(Data,1)),:);
plot(C1x, C1y, 'ro');
hold;
plot(C2x, C2y, 'bx');
title('Plotted data');
TrainData = [];
TrainData(1:218,1:2)=Data(1:218,1:2);
TestData = [];
TestData(1:110,1:2)=Data(219:328,1:2);
testlabels1=ones(size(TestData)/2);testlabels1=testlabels1';
testlabels2=-1*ones(size(TestData)/2);testlabels2=testlabels2';
TestLabels = [testlabels1 testlabels2];
trainlabels1=ones(size(TrainData)/2);trainlabels1=trainlabels1';
trainlabels2=-1*ones(size(TrainData)/2);trainlabels2=trainlabels2';
TrainLabels = [trainlabels1 trainlabels2];
labels1 = ones(size(Data)/2);labels1=labels1';
labels2 = -ones(size(Data)/2);labels2=labels2';
labels = [labels1 labels2];
%%fold1
[N,Dim]=size(TrainData);
H=zeros(length(TrainData));
len =length(TrainData);
for i=1:len
    for j=1:len
    if(Kcount==0)% Polynomial
        K=(TrainData(j,:)*TrainData(i,:)' + 1)^2;
    end
    if(Kcount==1)% Dot Product
        K=(TrainData(j,:)*TrainData(i,:)');
    end
        H(i,j) =TrainLabels(i)*TrainLabels(j)*K;
    end
end
ff=-ones(N,1);
Aeq=zeros(N,N);
Aeq(1,:)=TrainLabels;
beq=zeros(N,1);

[x,fval,exitflag,output,lambda]=quadprog(H+eye(N)*0.0001,ff,[],[],Aeq,beq,zeros(1,N),Cval*ones(1,N));
supportVectors = find(x > eps);
supportX = x(supportVectors);
supportData = TrainData(supportVectors,:);
supportLabels = TrainLabels(supportVectors);
supportLength =length(supportLabels);
% Now, solve for b Create a set of b's and average them
Bset =[];
for i=1:supportLength 
    Bval =0;
    for j=1:supportLength
        if(Kcount==0)
        K=(supportData(i,:)*supportData(j,:)' +1)^2;
        end
        if(Kcount==1)
        K=supportData(i,:)*supportData(j,:)';
        end
        Bval = Bval + ( supportX(j) * supportLabels(j) * K);
    end
    Bval = supportLabels(i) * Bval;
    Bval = (1 -Bval)/supportLabels(i); 
    Bset = [Bset Bval];
end
b =mean(Bset);
Res =zeros(1,N);
C1res =0;
C2res =0;
C1wrong=0;
C2wrong=0;
for i=1:size(TestData)
    sumVal=0;
    for j=1:supportLength
        if(Kcount==0)
        K=(supportData(j,:)*TestData(i,:)' +1)^2;
        end
        if(Kcount==1)
            K=supportData(j,:)*TestData(i,:)';
        end
        sumVal = sumVal +supportX(j)*supportLabels(j)*K;
    end
    Res(i) = sumVal +b;
    if(Res(i)>0 && TestLabels(i)>0 )
        C1res = C1res +1;
    end
    if(Res(i)<0 && TestLabels(i)<0)
       C2res = C2res +1; 
    end
    if(Res(i)>0 && TestLabels(i)<0)
       C1wrong= C1wrong+1; 
    end
    if(Res(i)<0 && TestLabels(i)>0)
       C2wrong= C2wrong+1; 
    end
    
end
C1cr = C1res/(C1res+C2wrong);
C2cr = C2res/(C2res+C1wrong);
%%fold2
TrainData = [];
TrainData(1:109,1:2)=Data(1:109,1:2);
TrainData(110:218,1:2)=Data(220:328,1:2);
TestData = [];
TestData(1:110,1:2)=Data(110:219,1:2);
testlabels1=ones(size(TestData)/2);testlabels1=testlabels1';
testlabels2=-1*ones(size(TestData)/2);testlabels2=testlabels2';
TestLabels = [testlabels1 testlabels2];
trainlabels1=ones(size(TrainData)/2);trainlabels1=trainlabels1';
trainlabels2=-1*ones(size(TrainData)/2);trainlabels2=trainlabels2';
TrainLabels = [trainlabels1 trainlabels2];
labels1 = ones(size(Data)/2);labels1=labels1';
labels2 = -ones(size(Data)/2);labels2=labels2';
labels = [labels1 labels2];

[N,Dim]=size(TrainData);
H=zeros(length(TrainData));
len =length(TrainData);
for i=1:len
    for j=1:len
    if(Kcount==0)% Polynomial
        K=(TrainData(j,:)*TrainData(i,:)' + 1)^2;
    end
    if(Kcount==1)% Dot Product
        K=(TrainData(j,:)*TrainData(i,:)');
    end
    H(i,j) =TrainLabels(i)*TrainLabels(j)*K;
    end
end
ff=-ones(N,1);
Aeq=zeros(N,N);
Aeq(1,:)=TrainLabels;
beq=zeros(N,1);

[x,fval,exitflag,output,lambda]=quadprog(H+eye(N)*0.0001,ff,[],[],Aeq,beq,zeros(1,N),Cval*ones(1,N));
supportVectors = find(x > eps);
supportX = x(supportVectors);
supportData = TrainData(supportVectors,:);
supportLabels = TrainLabels(supportVectors);
supportLength =length(supportLabels);
% Now, solve for b Create a set of b's and average them
Bset =[];
for i=1:supportLength 
    Bval =0;
    for j=1:supportLength
        if(Kcount==0)
        K=(supportData(i,:)*supportData(j,:)' +1)^2;
        end
        if(Kcount==1)
        K=supportData(i,:)*supportData(j,:)';
        end
        Bval = Bval + ( supportX(j) * supportLabels(j) * K);
    end
    Bval = supportLabels(i) * Bval;
    Bval = (1 -Bval)/supportLabels(i); 
    Bset = [Bset Bval];
end
b =mean(Bset);
Res =zeros(1,N);
f2C1res =0;
f2C2res =0;
f2C1wrong=0;
f2C2wrong=0;
for i=1:size(TestData)
    sumVal=0;
    for j=1:supportLength
        if(Kcount==0)
        K=(supportData(j,:)*TestData(i,:)' +1)^2;
        end
        if(Kcount==1)
            K=supportData(j,:)*TestData(i,:)';
        end
        sumVal = sumVal +supportX(j)*supportLabels(j)*K;
    end
    Res(i) = sumVal +b;
    if(Res(i)>0 && TestLabels(i)>0 )
        f2C1res = f2C1res +1;
    end
    if(Res(i)<0 && TestLabels(i)<0)
       f2C2res = f2C2res +1; 
    end
    if(Res(i)>0 && TestLabels(i)<0)
       f2C1wrong= f2C1wrong+1; 
    end
    if(Res(i)<0 && TestLabels(i)>0)
       f2C2wrong= f2C2wrong+1; 
    end
    
end
f2C1cr = f2C1res/(f2C1res+f2C2wrong);
f2C2cr = f2C2res/(f2C2res+f2C1wrong);
%%fold 3
TrainData = [];
TrainData(1:218,1:2)=Data(111:328,1:2);
TestData = [];
TestData(1:110,1:2)=Data(1:110,1:2);
testlabels1=ones(size(TestData)/2);testlabels1=testlabels1';
testlabels2=-1*ones(size(TestData)/2);testlabels2=testlabels2';
TestLabels = [testlabels1 testlabels2];
trainlabels1=ones(size(TrainData)/2);trainlabels1=trainlabels1';
trainlabels2=-1*ones(size(TrainData)/2);trainlabels2=trainlabels2';
TrainLabels = [trainlabels1 trainlabels2];
labels1 = ones(size(Data)/2);labels1=labels1';
labels2 = -ones(size(Data)/2);labels2=labels2';
labels = [labels1 labels2];

[N,Dim]=size(TrainData);
H=zeros(length(TrainData));
len =length(TrainData);
for i=1:len
    for j=1:len
    if(Kcount==0)% Polynomial
        K=(TrainData(j,:)*TrainData(i,:)' + 1)^2;
    end
    if(Kcount==1)% Dot Product
        K=(TrainData(j,:)*TrainData(i,:)');
    end
    H(i,j) =TrainLabels(i)*TrainLabels(j)*K;
    end
end
ff=-ones(N,1);
Aeq=zeros(N,N);
Aeq(1,:)=TrainLabels;
beq=zeros(N,1);

[x,fval,exitflag,output,lambda]=quadprog(H+eye(N)*0.0001,ff,[],[],Aeq,beq,zeros(1,N),Cval*ones(1,N));
supportVectors = find(x > eps);
supportX = x(supportVectors);
supportData = TrainData(supportVectors,:);
supportLabels = TrainLabels(supportVectors);
supportLength =length(supportLabels);
% Now, solve for b Create a set of b's and average them
Bset =[];
for i=1:supportLength 
    Bval =0;
    for j=1:supportLength
        if(Kcount==0)
        K=(supportData(i,:)*supportData(j,:)' +1)^2;
        end
        if(Kcount==1)
        K=supportData(i,:)*supportData(j,:)';
        end
        Bval = Bval + ( supportX(j) * supportLabels(j) * K);
    end
    Bval = supportLabels(i) * Bval;
    Bval = (1 -Bval)/supportLabels(i); 
    Bset = [Bset Bval];
end
b =mean(Bset);
Res =zeros(1,N);
f3C1res =0;
f3C2res =0;
f3C1wrong=0;
f3C2wrong=0;
for i=1:size(TestData)
    sumVal=0;
    for j=1:supportLength
        if(Kcount==0)
        K=(supportData(j,:)*TestData(i,:)' +1)^2;
        end
        if(Kcount==1)
            K=supportData(j,:)*TestData(i,:)';
        end
        sumVal = sumVal +supportX(j)*supportLabels(j)*K;
    end
    Res(i) = sumVal +b;
    if(Res(i)>0 && TestLabels(i)>0 )
        f3C1res = f3C1res +1;
    end
    if(Res(i)<0 && TestLabels(i)<0)
       f3C2res = f3C2res +1; 
    end
    if(Res(i)>0 && TestLabels(i)<0)
       f3C1wrong= f3C1wrong+1; 
    end
    if(Res(i)<0 && TestLabels(i)>0)
       f3C2wrong= f3C2wrong+1; 
    end
    
end
f3C1cr = f3C1res/(f3C1res+f3C2wrong);
f3C2cr = f3C2res/(f3C2res+f3C1wrong);
CR = 100*((C1cr+f2C1cr+f3C1cr+C2cr+f2C2cr+f3C2cr)/6);

Ktype='';
if(Kcount==0)
    Ktype='Polynomial';
end
if(Kcount==1)
    Ktype='Dot Product';
end
fprintf('Kernel: %s  C value: %i\n',Ktype,Cval);
fprintf('Classification Rate: %f\n',CR);
end
end