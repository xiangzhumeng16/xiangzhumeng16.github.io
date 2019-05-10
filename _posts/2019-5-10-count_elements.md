Counting the number of occurrences of elements in Matlab.
```matlab
len = 100;
iMax = 10;
X = randi(iMax, len, 1);
count = hist(X,unique(X));
disp(count);
```
