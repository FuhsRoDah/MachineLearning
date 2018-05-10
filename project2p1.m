fprintf('PCA Graphs\n')
X = readtable('wine.xls');
X = table2array(X);
[EIGVEC, EIGVAL] = eig(cov(X));
XT = [];
for i=1:size(X)
	XT(i,1) = X(i,:) * EIGVEC(:,13);
end  
hist(XT,200),pause

XT = [];
for i=1:size(X)
	XT(i,1) = X(i,:) * EIGVEC(:,13);
	XT(i,2) = X(i,:) * EIGVEC(:,12);
end  
plot(XT(:,1),XT(:,2),'or'), pause

XT = [];
for i=1:size(X)
	XT(i,1) = X(i,:) * EIGVEC(:,13);
	XT(i,2) = X(i,:) * EIGVEC(:,12);
	XT(i,3) = X(i,:) * EIGVEC(:,11);
end  
scatter3(XT(:,1),XT(:,2),XT(:,3)), pause
fprintf('LDA Graphs\n')
