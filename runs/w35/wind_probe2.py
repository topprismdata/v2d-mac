import sys
import numpy as np
sys.path.insert(0, '.')
import mujoco
from retarget import retarget_smoke as rs

mj, model, data = rs.load_scene()
rs._mj_forward(model, data)
seq = rs.load_source_sequence()
joints21, _ = rs.mano_skeleton(seq)
retargeter, _ = rs.build_retargeter()
qpos = rs.retarget_sequence(joints21, retargeter)
anchor = rs.executed_anchor(qpos, model, data)
f_act = np.arange(6, 22)
w_act = np.arange(6)
lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
gf = np.clip(anchor["finger_cmd"][0], lo[f_act], hi[f_act])
gw = np.clip(anchor["ctrl"][0], lo[w_act], hi[w_act])
fq = rs._finger_qadr(model)
wq = np.array([model.jnt_qposadr[rs._joint_id(model, n)] for n in rs.WRIST_JOINTS])
frame_skip = max(1, int(round(1.0 / (rs.CTRL_HZ * model.opt.timestep))))
rs.reset_scene(model, data, fq, wq, f_act, w_act)
rs.warm_start(model, data, fq, wq, f_act, w_act, gf, gw,
              ctrl_hz=rs.CTRL_HZ, frame_skip=frame_skip)
for _ in range(frame_skip):
    mujoco.mj_step(model, data)

M = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, data, M)
names = rs.MJCF_FINGER_JOINTS
for nm in ("mfj1", "mfj0"):
    jid = rs._joint_id(model, nm)
    adr = int(model.jnt_dofadr[jid])
    qadr = int(model.jnt_qposadr[jid])
    a = rs._joint_actuator_id(model, nm)
    kp = float(model.actuator_gainprm[a, 0])
    tot = (-data.qfrc_bias[adr] + data.qfrc_passive[adr]
           + data.qfrc_constraint[adr] + data.qfrc_actuator[adr]
           + data.qfrc_applied[adr])
    other = -data.qfrc_bias[adr] + data.qfrc_passive[adr] + data.qfrc_constraint[adr]
    print(f"{nm} adr={adr} qpos={data.qpos[qadr]:.6f} ctrl={data.ctrl[a]:.6f}")
    print(f"  kp={kp} M_diag={M[adr,adr]:.6e} dof_M0={model.dof_M0[adr]:.6e} "
          f"arm={model.dof_armature[adr]:.3e}")
    print(f"  passive={data.qfrc_passive[adr]:+.8f} bias={data.qfrc_bias[adr]:+.8f} "
          f"constraint={data.qfrc_constraint[adr]:+.8f} actuator={data.qfrc_actuator[adr]:+.8f} "
          f"applied={data.qfrc_applied[adr]:+.8f}")
    print(f"  sum={tot:+.8f} M*qacc={M[adr,adr]*data.qacc[adr]:+.8f} qacc={data.qacc[adr]:+.6f}")
    print(f"  |actuator|={abs(data.qfrc_actuator[adr]):.8f} |constraint|={abs(data.qfrc_constraint[adr]):.8f} "
          f"ratio={abs(data.qfrc_actuator[adr])/max(abs(data.qfrc_constraint[adr]),1e-12):.4f}")
    print(f"  tau_other={other:+.8f} |tau_servo|={abs(data.qfrc_actuator[adr]):.8f} rel={abs(abs(data.qfrc_actuator[adr])-abs(other))/max(abs(other),1e-15):.5f}")
