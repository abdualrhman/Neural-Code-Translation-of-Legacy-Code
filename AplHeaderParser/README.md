## AplHeaderParser

to parse an apl header

```
./linux-x64/AplHeaderParser <APL_HEADER> <OUTOUT_METHOD_NAME>
```

linux example

```
./linux-x64/AplHeaderParser_linux "⍝V: V=, R=-0, M=1
⍝0: Delete leading blanks on many texts (xDlb each).
⍝3: x As {vtSTRING}|vtSTRING[]|vtSTRING[;]                   : vector or matrix of text vectors
⍝4: r As {vtSTRING}|vtSTRING[$3:x:1]|vtSTRING[$3:x:1;$3:x:2] : vector or matrix of stripped text vectors
⍝5: AFC: RC{($3:x:0,$4:r:0)|($3:x:1,$4:r:1)|($3:x:2,$4:r:2)}
⍝5: FST: StaticTypeChk" "DeleteLeadingBlanks"
```

output

```
public string DeleteLeadingBlanks(string x)
public string DeleteLeadingBlanks(string[] x)
public string DeleteLeadingBlanks(string[,] x)
public string[] DeleteLeadingBlanks1d(string x)
public string[] DeleteLeadingBlanks1d(string[] x)
public string[] DeleteLeadingBlanks1d(string[,] x)
public string[,] DeleteLeadingBlanks2d(string x)
public string[,] DeleteLeadingBlanks2d(string[] x)
public string[,] DeleteLeadingBlanks2d(string[,] x)
```
