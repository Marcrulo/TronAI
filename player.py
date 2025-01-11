
class Player:
    def __init__(self, start_pos, start_dir, scale, width, height):
        self.x, self.y = start_pos
        self.newx, self.newy = self.x, self.y
        self.dir = start_dir
        self.trail = []
        self.alive = True
        
        self.scale = scale
        self.width = width
        self.height = height
    
    def kill(self):
        self.alive = False
        
    def update(self):
        if not self.alive:
            return 
        
        if self.dir == 'right':
            self.newx = self.x + self.scale
            
        elif self.dir == 'left':
            self.newx = self.x - self.scale
            
        elif self.dir == 'down':
            self.newy = self.y + self.scale
            
        elif self.dir == 'up':
            self.newy = self.y - self.scale
        
        # collision with itself
        for t in self.trail:
            if self.x == t[0] and self.y == t[1]:
                self.kill()
                print("COLLISION WITH SELF")
                return
                
        # collision with border
        if self.newx < self.scale:
            self.newx = self.x
            self.kill()
            print("collision with left border")
            return
        
        elif self.newx > self.width:
            self.newx = self.x
            self.kill()
            print("collision with right border") 
            return
           
        elif self.newy < self.scale:
            self.newy = self.y
            self.kill()
            print("collision with top border") 
            return
        
        elif self.newy > self.height:
            self.newy = self.y
            self.kill()
            print("collision with bottom border") 
            return
                
        # add to trail
        self.trail.append((self.x, self.y))
        
        # update position
        self.x, self.y = self.newx, self.newy