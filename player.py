
class Player:
    def __init__(self, start_pos, start_dir):
        self.x, self.y = start_pos
        self.newx, self.newy = self.x, self.y
        self.dir = start_dir
        self.trail = []
        self.alive = True
    
    def kill(self):
        self.alive = False
        
    def update(self, scale, width, height, p_opponent):
        if not self.alive:
            return 
        
        if self.dir == 'right':
            self.newx = self.x + scale
            
        elif self.dir == 'left':
            self.newx = self.x - scale
            
        elif self.dir == 'down':
            self.newy = self.y + scale
            
        elif self.dir == 'up':
            self.newy = self.y - scale
        
        # collision with itself
        for t in self.trail:
            if self.x == t[0] and self.y == t[1]:
                self.kill()
                # print("collision with self")
                return
                
        # collision with border
        if self.newx < 1:
            self.newx = self.x
            self.kill()
            print("collision with left border")
            return
        
        elif self.newx > width-3*scale-1:
            self.newx = self.x
            self.kill()
            print("collision with right border") 
            return
           
        elif self.newy < 1:
            self.newy = self.y
            self.kill()
            print("collision with top border") 
            return
        
        elif self.newy > height-3*scale-1:
            self.newy = self.y
            self.kill()
            print("collision with bottom border") 
            return
        
        # collision with other player
        if self.x == p_opponent.x and self.y == p_opponent.y:
            self.kill()
            # print("collision with opponent")
            return
        
        for t in p_opponent.trail:
            if self.x == t[0] and self.y == t[1]:
                self.kill()
                # print("collision with opponent")
                return
                
                
        # add to trail
        self.trail.append((self.x, self.y))
        
        # update position
        self.x, self.y = self.newx, self.newy