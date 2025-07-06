// 4-bit Ripple Carry Adder Example
// This demonstrates HDL compilation in the FPGA simulator

module half_adder(sum, carry, a, b);
    output sum, carry;
    input a, b;
    
    xor g1 (sum, a, b);
    and g2 (carry, a, b);
endmodule

module full_adder(sum, cout, a, b, cin);
    output sum, cout;
    input a, b, cin;
    
    wire s1, c1, c2;
    
    // First half adder
    xor g1 (s1, a, b);
    and g2 (c1, a, b);
    
    // Second half adder
    xor g3 (sum, s1, cin);
    and g4 (c2, s1, cin);
    
    // Carry out
    or g5 (cout, c1, c2);
endmodule

module ripple_carry_adder_4bit(sum, cout, a, b, cin);
    output [3:0] sum;
    output cout;
    input [3:0] a, b;
    input cin;
    
    wire c1, c2, c3;
    
    // Cascade full adders
    full_adder fa0 (sum[0], c1, a[0], b[0], cin);
    full_adder fa1 (sum[1], c2, a[1], b[1], c1);
    full_adder fa2 (sum[2], c3, a[2], b[2], c2);
    full_adder fa3 (sum[3], cout, a[3], b[3], c3);
endmodule

// Test bench example (for simulation reference)
module testbench;
    reg [3:0] a, b;
    reg cin;
    wire [3:0] sum;
    wire cout;
    
    ripple_carry_adder_4bit adder(sum, cout, a, b, cin);
    
    initial begin
        // Test case 1: 5 + 3 = 8
        a = 4'b0101;
        b = 4'b0011;
        cin = 0;
        
        // Test case 2: 15 + 1 = 16 (overflow)
        #10 a = 4'b1111;
        b = 4'b0001;
        cin = 0;
        
        // Test case 3: 7 + 7 + 1 = 15
        #10 a = 4'b0111;
        b = 4'b0111;
        cin = 1;
    end
endmodule