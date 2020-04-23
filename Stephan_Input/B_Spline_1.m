function [ b, db_dx, d2b_dx2 ] = B_Spline_1( x, x_i, x_ip1, x_ip2 )
% Linearer Spline (Hut-Funktion)
    b       = 0;
    db_dx	= 0;
    d2b_dx2	= 0;
    
    if ( x_i == x_ip1 )
    % Erste Basisfunktion
        if x >= x_ip1 && x < x_ip2
            b       = ( x_ip2 - x )/( x_ip2 - x_ip1 );
            db_dx	= -1/( x_ip2 - x_ip1 );
            d2b_dx2	= 0;  
        end
    elseif ( x_ip1 == x_ip2 )
    % Letzte Basisfunktion
        if x >= x_i && x <= x_ip1
            b       = ( x - x_i )/( x_ip1 - x_i );
            db_dx	= 1/( x_ip1 - x_i );
            d2b_dx2	= 0;  
        end    
    else  
        if x >= x_i && x < x_ip1
            b       = ( x - x_i )/( x_ip1 - x_i );
            db_dx	= 1/( x_ip1 - x_i );
            d2b_dx2	= 0;
        elseif  x >= x_ip1 && x < x_ip2
            b       = ( x_ip2 - x )/( x_ip2 - x_ip1 );
            db_dx	= -1/( x_ip2 - x_ip1 );
            d2b_dx2	= 0;      
        end
    end
end

