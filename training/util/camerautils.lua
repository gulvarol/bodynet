-- Returns intrinsic camera matrix
-- Parameters are hard-coded since all SURREAL images use the same.
function getIntrinsicBlender()
  --These are set in Blender (datageneration/main_part1.py)
  local res_x_px         = 320 -- *scn.render.resolution_x
  local res_y_px         = 240 -- *scn.render.resolution_y
  local f_mm             = 60  -- *cam_ob.data.lens
  local sensor_w_mm      = 32  -- *cam_ob.data.sensor_width
  local sensor_h_mm = sensor_w_mm * res_y_px / res_x_px -- *cam_ob.data.sensor_height (function of others)

  local scale = 1 -- *scn.render.resolution_percentage/100
  local skew  = 0 -- only use rectangular pixels
  local pixel_aspect_ratio = 1

  -- From similar triangles:
  -- sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
  local fx_px = f_mm * res_x_px * scale / sensor_w_mm
  local fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

  -- Center of the image
  local u = res_x_px * scale / 2-- + 1 -- + 1 because we add +1 to the joints2D when loading (they are 0-indexed in Python)
  local v = res_y_px * scale / 2-- + 1

  -- Intrinsic camera matrix
  local K = torch.Tensor({ {fx_px, skew, u}, {0, fy_px, v}, {0, 0, 1} })
        
  return K
end

function getIntrinsicH36M()
  return  torch.Tensor({ {1148, 0, 508}, {0, 1148, 508}, {0, 0, 1} })
  --return  torch.Tensor({ {1148, 0, 160}, {0, 1148, 120}, {0, 0, 1} })
  --return  torch.Tensor({ {600, 0, 160}, {0, 600, 120}, {0, 0, 1} })
end

function getIntrinsicUP(flength, w, h)
  -- flength
  return  torch.Tensor({ {flength, 0, w/2}, {0, flength, h/2}, {0, 0, 1} })
  --return  torch.Tensor({ {flength, 0, 190}, {0, flength, 190}, {0, 0, 1} })
end


-- Returns extrinsic camera matrix  
--   T : translation vector from Blender (*cam_ob.location)
--   RT: extrinsic computer vision camera matrix 
--   Script based on https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
function getExtrinsicBlender(T)
  -- Take the first 3 columns of the matrix_world in Blender and transpose.
  -- This is hard-coded since all images in SURREAL use the same.
  R_world2bcam = torch.Tensor({ {0, 0, 1}, {0, -1, 0}, {-1, 0, 0} }):t()               
  -- *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
  --                               (0., -1, 0., -1.0),
  --                               (-1., 0., 0., 0.),
  --                               (0.0, 0.0, 0.0, 1.0)))

  -- Convert camera location to translation vector used in coordinate changes
  T_world2bcam = -1 * R_world2bcam * T

  -- Following is needed to convert Blender camera to computer vision camera
  R_bcam2cv = torch.Tensor({ {1, 0, 0}, {0, -1, 0}, {0, 0, -1} }) 

  -- Build the coordinate transform matrix from world to computer vision camera
  R_world2cv = R_bcam2cv*R_world2bcam
  T_world2cv = R_bcam2cv*T_world2bcam

  -- Put into 3x4 matrix
  RT = torch.cat(R_world2cv, T_world2cv, 2)

  return RT, R_world2cv, T_world2cv
end

function reconstruct3D(j2d, Kinv, T)
    local homcoords2D = torch.cat(j2d:t(), torch.ones(1, #opt.jointsIx), 1)
    local rec3D = (Kinv*homcoords2D):t()--- + T -- (2D - c)/f
    local z = T[{{}, {3}}]

    rec3D[{{}, {1}}] = torch.cmul(rec3D[{{}, {1}}], z)
    rec3D[{{}, {2}}] = torch.cmul(rec3D[{{}, {2}}], z)
    rec3D[{{}, {3}}] = z

    return rec3D

    --local homcoords2D = torch.cat(j2d:t(), z, 1)
    --return Kinv*homcoords2D
end

function bsxfunsum(mat, vec)
    local out = torch.Tensor():resizeAs(mat)
    for i = 1, mat:size(1) do
        out[i] = mat[i] + vec
    end
    return out
end