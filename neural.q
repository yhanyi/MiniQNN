/// Q Neural Network

// Activations
sigmoid:{1%(1+exp neg x)};
sigmoid_derivative:{x*(1-x)};

// Initialise
init:{[structure]
  if[2>count structure; '"Neural network must have at least input and output layers"];
  
  weights:();
  biases:();
    
  i:0;
  while[i<(count structure)-1;
    // Random weights between -1 and 1
    w:(structure[i+1];structure[i])#(2*structure[i+1]*structure[i])?1.0;
    weights,:enlist w;
        
    // Random biases between -1 and 1
    b:(structure[i+1])#(structure[i+1])?1.0;
    biases,:enlist b;
        
    i+:1;
    ];
    
  // Return a dictionary with weights and biases
  `weights`biases!(weights;biases)
  };

/// Test with XOR
example_x:(0 0; 0 1; 1 0; 1 1);
example_y:(0; 1; 1; 0);

nn:init[2 3 1];
show nn;
