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

// Forward Propagation
fwdprop:{[nn; x]
  // If x is a single sample, wrap in a list
  multiple:0h=type x;
  if[not multiple; x:enlist x];
  
  results:();
  j:0;
  
  // Process each sample
  while[j<count x;
    sample:x[j];
    activations:enlist sample;
    
    // Process each layer
    i:0;
    while[i<count nn[`weights];
      // Get input for this layer
      input:last activations;
      
      // Get weights and biases for this layer
      w:nn[`weights][i];
      b:nn[`biases][i];
      
      // Calculate weighted sum for this layer
      z:();
      n:0;
      while[n<count w;
        // Calculate weighted sum for this neuron
        weighted_sum:sum input * w[n];
        
        // Add bias
        weighted_sum:weighted_sum + b[n];
        
        // Add to layer output
        z,:enlist weighted_sum;
        n+:1;
        ];
      
      // Apply activation function
      a:sigmoid z;
      
      // Add to activations
      activations,:enlist a;
      
      // Move to next layer
      i+:1;
      ];
    
    // Store final result for this sample
    results,:enlist last activations;
    
    // Move to next sample
    j+:1;
    ];
  
  // If input x was a single sample, return a single result, else return list of results
  $[multiple; results; first results]
  };


/// Test with XOR
example_x:(0 0; 0 1; 1 0; 1 1);
example_y:(0; 1; 1; 0);

nn:init[2 3 1];
show nn;

activations:fwdprop[nn; example_x];
show activations;
