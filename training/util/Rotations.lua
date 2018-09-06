local Rotations = {}

local function normalize(v)
   local v_mag = torch.norm(v)
   if v_mag == 0 then
      v = torch.zeros(3)
      v[1] = 1;
   else
      v = v / v_mag;
   end
   return v
end

-- Rodrigues formula
local function rotvec2rotmat(rotvec) -- gets a 3x1 rotation vector (e.g. pose)
   local theta = torch.norm(rotvec);
   local r
   if theta > 0 then
       r = rotvec/theta;
   else
       r = rotvec;
   end
   local cost = torch.cos(theta);
   local mat = torch.Tensor({ {    0,  -r[3]:squeeze(),   r[2]:squeeze()},
                        { r[3]:squeeze(),      0,  -r[1]:squeeze()},
                        {-r[2]:squeeze(),   r[1]:squeeze(),     0 }  })
   return (cost*torch.eye(3) + (1-cost)* r*(r:t()) + torch.sin(theta)*mat);
end

local function rotmat2euler(rotmat)
   local sy = math.sqrt(rotmat[1][1] * rotmat[1][1] + rotmat[2][1] * rotmat[2][1])
   local roteulerx = torch.atan2(rotmat[3][2], rotmat[3][3]) -- x rotation in Euler
   local roteulery = torch.atan2(-rotmat[3][1], sy)          -- y rotation in Euler
   local roteulerz = torch.atan2(rotmat[2][1], rotmat[1][1]) -- z rotation in Euler
   local roteuler = torch.Tensor({roteulerx, roteulery, roteulerz})
   return roteuler
end

-- Convert rotation matrix representation to axis-angle (rotation vector) representation
-- R: Rotation matrix, u: unit vector, theta: angle
-- https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis.E2.80.93angle
-- https://github.com/tuckermcclure/vector-and-rotation-toolbox/blob/master/dcm2aa.m
local function rotmat2rotvec(R)
   local u = torch.zeros(3)
   local x = 0.5*(R[1][1] + R[2][2] + R[3][3] - 1) --1.0000000484288
   x = math.max(x, -1)
   x = math.min(x, 1)
   local theta = math.acos(x) -- Tr(R) = 1 + 2 cos(theta)

   if(theta < 1e-4) then -- avoid division by zero!
      --u[1] = 1
      print('theta ~= 0')
      return u
   elseif(math.abs(theta - math.pi) < 1e-4) then
      print('theta ~= pi')
      if (R[1][1] >= R[3][3]) then
         if (R[1][1] >= R[2][2]) then
            u[1] = R[1][1] + 1
            u[2] = R[2][1]
            u[3] = R[3][1]
         else
            u[1] = R[1][2]
            u[2] = R[2][2] + 1
            u[3] = R[3][2]
         end
      else
         u[1] = R[1][3]
         u[2] = R[2][3]
         u[3] = R[3][3] + 1
      end

      u = normalize(u)
   else
      local d = 1/(2 * math.sin(theta)) --||u|| = 2sin(theta)
      u[1] = d * (R[3][2] - R[2][3])
      u[2] = d * (R[1][3] - R[3][1])
      u[3] = d * (R[2][1] - R[1][2])
   end
   return u*theta
end

Rotations.rotvec2rotmat     = rotvec2rotmat
Rotations.rotmat2euler      = rotmat2euler
Rotations.rotmat2rotvec     = rotmat2rotvec

return Rotations