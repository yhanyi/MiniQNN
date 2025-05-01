/// Mini Q Neural Network

sig:{1%(1+exp neg x)};
sigd:{x*(1-x)};
shape:{(count x;count last x)}
init:{flip flip[r]-avg r:{[x;y]x?1.0}[y]each til x};
fwd:{[x;y;lr;d]
  z:1.0,/:sig[x mmu d`w];
  o:sig[z mmu d`v];
  do:y-o;
  dh:1_/:$[do;flip d`v]*sigd z;
  `o`v`w!(o;d[`v]+lr*flip[z] mmu do;d[`w]+lr*flip[x] mmu dh)
  };

qns:(`XOR`AND`OR`NAND)!(
  (0 1 1 0f);   // XOR
  (0 0 0 1f);   // AND
  (0 1 1 1f);   // OR
  (1 1 1 0f)    // NAND
  );

x:((0 0f);(0 1f);(1 0f);(1 1f))
x:x,'1.0;

{[t;y]
  -1 "\n= Test ",string[t]," =";
  w:init . (3;count first x);
  v:init . (1;4);
  res:(fwd[x;y;0.1]/)[10000;`o`w`v!(0,();w;v)];
  -1 type string[y];
  -1 string[y];
  -1 string[res`o];
  }[;]each (key qns;value qns);
