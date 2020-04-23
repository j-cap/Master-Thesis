function [ b, db_dx, d2b_dx2 ] = B_Spline_2( x, x_i, x_ip1, x_ip2, x_ip3 )
% Quadratischer Spline
    b       = 0;
    db_dx	= 0;
    d2b_dx2	= 0;
    
    if ( x_i == x_ip1 ) && ( x_ip1 == x_ip2 )
    % Erste Basisfunktion
        if x >= x_ip2 && x < x_ip3
        	b       = ( x_ip3 - x )^2/( ( x_ip3 - x_ip1 )*( x_ip3 - x_ip2 ) );
            db_dx	= -2*( x_ip3 - x )/( ( x_ip3 - x_ip1 )*( x_ip3 - x_ip2 ) );            
            d2b_dx2	= 2/( ( x_ip3 - x_ip1 )*( x_ip3 - x_ip2 ) ); 
        end    
    elseif ( x_i == x_ip1 )
    % Zweite Basisfunktion
        if x >= x_ip1 && x < x_ip2 
            b       =  ( ( x - x_i )*( x_ip2 - x ) )/( ( x_ip2 - x_i )*( x_ip2 - x_ip1 ) ) + ( ( x_ip3 - x )*( x - x_ip1 ) )/( ( x_ip3 - x_ip1 )*( x_ip2 - x_ip1 ) );
            db_dx	=  ( x_ip2 - 2*x + x_i )/( ( x_ip2 - x_i )*( x_ip2 - x_ip1 ) ) + ( x_ip1 + x_ip3 - 2*x )/( ( x_ip3 - x_ip1 )*( x_ip2 - x_ip1 ) );
            d2b_dx2	= - 2/( ( x_ip2 - x_i )*( x_ip2 - x_ip1 ) ) + - 2/( ( x_ip3 - x_ip1 )*( x_ip2 - x_ip1 ) );  
        elseif x >= x_ip2 && x < x_ip3
            b       = ( x_ip3 - x )^2/( ( x_ip3 - x_ip1 )*( x_ip3 - x_ip2 ) );      
            db_dx	= -2*( x_ip3 - x )/( ( x_ip3 - x_ip1 )*( x_ip3 - x_ip2 ) ); 
            d2b_dx2	= 2/( ( x_ip3 - x_ip1 )*( x_ip3 - x_ip2 ) ); 
        end       
    elseif ( x_ip1 == x_ip2 ) && ( x_ip2 == x_ip3 )
    % Letzte Basisfunktion    
        if x >= x_i && x <= x_ip1
            b       = ( x - x_i )^2/( ( x_ip2 - x_i)*( x_ip1 - x_i) ); 
            db_dx	= 2*( x - x_i )/( ( x_ip2 - x_i)*( x_ip1 - x_i) );
            d2b_dx2	= 2/( ( x_ip2 - x_i)*( x_ip1 - x_i) );
        end     
    elseif ( x_ip2 == x_ip3 )
    % Vorletzte Basisfunktion    
        if x >= x_i && x < x_ip1
            b       = ( x - x_i )^2/( ( x_ip2 - x_i)*( x_ip1 - x_i) );
            db_dx	= 2*( x - x_i )/( ( x_ip2 - x_i)*( x_ip1 - x_i) );
            d2b_dx2	= 2/( ( x_ip2 - x_i)*( x_ip1 - x_i) );                     
        elseif x >= x_ip1 && x <= x_ip2
            b       =  ( ( x - x_i )*( x_ip2 - x ) )/( ( x_ip2 - x_i )*( x_ip2 - x_ip1 ) ) + ( ( x_ip3 - x )*( x - x_ip1 ) )/( ( x_ip3 - x_ip1 )*( x_ip2 - x_ip1 ) );
            db_dx	=  ( x_ip2 - 2*x + x_i )/( ( x_ip2 - x_i )*( x_ip2 - x_ip1 ) ) + ( x_ip1 + x_ip3 - 2*x )/( ( x_ip3 - x_ip1 )*( x_ip2 - x_ip1 ) );
            d2b_dx2	= - 2/( ( x_ip2 - x_i )*( x_ip2 - x_ip1 ) ) + - 2/( ( x_ip3 - x_ip1 )*( x_ip2 - x_ip1 ) );  
        end        
    else        
        if x >= x_i && x < x_ip1
            b       = ( x - x_i )^2/( ( x_ip2 - x_i)*( x_ip1 - x_i) );
            db_dx	= 2*( x - x_i )/( ( x_ip2 - x_i)*( x_ip1 - x_i) );
            d2b_dx2	= 2/( ( x_ip2 - x_i)*( x_ip1 - x_i) );            
        elseif x >= x_ip1 && x < x_ip2
            b       =  ( ( x - x_i )*( x_ip2 - x ) )/( ( x_ip2 - x_i )*( x_ip2 - x_ip1 ) ) + ( ( x_ip3 - x )*( x - x_ip1 ) )/( ( x_ip3 - x_ip1 )*( x_ip2 - x_ip1 ) );
            db_dx	=  ( x_ip2 - 2*x + x_i )/( ( x_ip2 - x_i )*( x_ip2 - x_ip1 ) ) + ( x_ip1 + x_ip3 - 2*x )/( ( x_ip3 - x_ip1 )*( x_ip2 - x_ip1 ) );
            d2b_dx2	= - 2/( ( x_ip2 - x_i )*( x_ip2 - x_ip1 ) ) + - 2/( ( x_ip3 - x_ip1 )*( x_ip2 - x_ip1 ) );  
        elseif x >= x_ip2 && x < x_ip3
            b       = ( x_ip3 - x )^2/( ( x_ip3 - x_ip1 )*( x_ip3 - x_ip2 ) ); 
            db_dx	= -2*( x_ip3 - x )/( ( x_ip3 - x_ip1 )*( x_ip3 - x_ip2 ) );
            d2b_dx2	= 2/( ( x_ip3 - x_ip1 )*( x_ip3 - x_ip2 ) ); 
        end  
    end
end

