// Source: https://github.com/Ruochun/pyDEME_tests/blob/main/DFCModel.cu
// DFC fresh-concrete contact model used by pyDEME_DFCSlump.py.
if (overlapDepth > 0) {
    // Get material properties
    //
    float mortar_layer = mortar_layer_mat[bodyAMatType];
    float ENm = ENm_mat[bodyAMatType];
    float ENa = ENa_mat[bodyAMatType];
    float alpha = alpha_mat[bodyAMatType];
    float beta = beta_mat[bodyAMatType];
    float np = np_mat[bodyAMatType];
    float sgmTmax = sgmTmax_mat[bodyAMatType];
    float sgmTau0 = sgmTau0_mat[bodyAMatType];
    float kappa0 = kappa0_mat[bodyAMatType];
    float eta_inf = eta_inf_mat[bodyAMatType];
    float flocbeta = flocbeta_mat[bodyAMatType];
    float flocm = flocm_mat[bodyAMatType];
    float flocTcr = flocTcr_mat[bodyAMatType];

    //
    float h = mortar_layer;
    float R1 = 5000.0;
    float R2 = 5000.0;
    double Rmin, Rmax, delta_rmin;
    double Lij, L0ij;
    // Material parameters of viscous part which should be defined inside
    // contactinfo
    double eta0 = kappa0 * eta_inf;
    double Deps0 = sgmTau0 / eta0;

    //
    // Get state variables relevant to this contact
    //
    float3 statevar;
    statevar.x = contact_info_strain_x;
    statevar.y = contact_info_strain_y;
    statevar.z = contact_info_strain_z;
    float lambda0 = contact_info_lambda;
    //
    // ==== SIGMA-t influenced by Flocculation ======
    sgmTmax = sgmTmax * (1. + lambda0);
    //
    if (lambda0 == 0.) {
        lambda0 = lambda_init_mat[bodyAMatType];
        float val0 = ParticleThixoAvg[AOwner];
        float val1 = ParticleThixoAvg[BOwner];
        //
        lambda0 = DEME_MIN(val0, val1);
        contact_info_lambda = lambda0;
        // std::cout << "lambda-4 " << map_contact_info[mykey].lambda <<  "key "
        // << mykey << std::endl;
        contact_info_step_time = time;
    }
    //
    // initialize force values
    //
    force = make_float3(0, 0, 0);

    float3 Vx, Vy, Vz;
    float3 normal_dir = -B2A;  // DEME uses B2A as normal direction, opposite
                               // of Chrono as it seems

    XdirToDxDyDz(normal_dir, make_float3(0, 1, 0), Vx, Vy, Vz);
    // std::cout<<" Vy: "<<Vy<<" Vz: "<<Vz<<"\t";

    double delta = overlapDepth;

    R1 = ARadius;
    R2 = BRadius;

    if (ContactType == deme::SPHERE_SPHERE_CONTACT) {
        // Concrete - Concrete Interaction
        // Contact of two DFC spheres
        // center to center distance between 2 objects
        L0ij = R1 + R2 - h;
        Lij = abs(R1 + R2 - delta);
        Rmin = DEME_MIN(R1, R2);
        Rmax = DEME_MAX(R1, R2);
        delta_rmin = delta / 2 * (2.0 * Rmax - delta) / Lij;

    } else if (ContactType == deme::SPHERE_MESH_CONTACT) {
        // Concrete-container interaction
        R2 = 5.;  //  0.;
        h = mortar_layer / 2.;
        // sgmTmax=this->material->Get_sgmTmax()/2;
        L0ij = R1 + R2 - h;
        Lij = abs(R1 + R2 - delta);
        Rmin = R1;
        // Rmin=std::min(R1, R2);
        Rmax = DEME_MAX(R1, R2);
        delta_rmin = delta;

    } else {
        /// Concrete-analytical interaction
        R2 = 6.;  // 0.;
        h = mortar_layer / 2.;
        // sgmTmax=this->material->Get_sgmTmax()/2;
        L0ij = R1 + R2 - h;
        Lij = abs(R1 + R2 - delta);
        Rmin = R1;
        Rmax = DEME_MAX(R1, R2);
        delta_rmin = delta;
    }

    double ai = abs(Rmin - delta_rmin);
    double radius2 = Rmin * Rmin - ai * ai;
    double contact_area = deme::PI * radius2;

    //
    // modify penetration according to initial overlap
    //
    double delta_new = delta - h;
    //
    //
    // Relative velocities at contact at midplane (p0=(p1+p2)/2)
    //
    // In DEME, contact point in global is just called contactPnt

    // In DEME, local contact point is called locCPA and locCPB
    // DEME uses its own way to calculate rel vel
    float3 rotVelCPA, rotVelCPB;
    {
        // We also need the relative velocity between A and B in global frame to
        // use in the damping terms To get that, we need contact points'
        // rotational velocity in GLOBAL frame This is local rotational velocity
        // (the portion of linear vel contributed by rotation)
        rotVelCPA = cross(ARotVel, locCPA);
        rotVelCPB = cross(BRotVel, locCPB);
        // This is mapping from local rotational velocity to global
        applyOriQToVector3<float, deme::oriQ_t>(rotVelCPA.x, rotVelCPA.y, rotVelCPA.z, AOriQ.w, AOriQ.x, AOriQ.y,
                                                AOriQ.z);
        applyOriQToVector3<float, deme::oriQ_t>(rotVelCPB.x, rotVelCPB.y, rotVelCPB.z, BOriQ.w, BOriQ.x, BOriQ.y,
                                                BOriQ.z);
    }
    float time_scale_val = 1.;
    float3 relvel = (BLinVel + rotVelCPB) - (ALinVel + rotVelCPA);  // A2B vel
    relvel = relvel / time_scale_val;

    float relvel_n_mag = dot(relvel, normal_dir);
    float3 relvel_n = relvel_n_mag * normal_dir;
    float3 relvel_t = relvel - relvel_n;
    float relvel_t_mag = length(relvel_t);
    //
    // Calculate displacement increment in normal and tangential direction
    //
    float dT = ts;
    float delta_t = relvel_t_mag * dT;
    float delta_n = relvel_n_mag * dT;
    float3 v_delta_t = relvel_t * dT;
    //
    //  Calculate the strain increment in each local direction
    //
    float depsN = delta_n / Lij;
    float delta_M = dot(relvel, Vy) * dT;  // v_delta_t.Dot(Vy);
    float delta_L = dot(relvel, Vz) * dT;  // v_delta_t.Dot(Vz);
    float depsM = delta_M / Lij;
    float depsL = delta_L / Lij;
    //
    //
    //
    float epsA = log(1 - h / (L0ij));
    float epsN = log(Lij / L0ij);

    // float epsN=statevar.x+depsN;
    float epsM = statevar.y + depsM;
    float epsL = statevar.z + depsL;
    //
    float epsT = powf((epsM * epsM + epsL * epsL), 0.5);
    float epsQ = powf(epsN * epsN + alpha * epsT * epsT, 0.5);
    //
    //
    //
    statevar.x = epsN;
    statevar.y = epsM;
    statevar.z = epsL;
    contact_info_strain_x = statevar.x;
    contact_info_strain_y = statevar.y;
    contact_info_strain_z = statevar.z;
    contact_info_step_time = time;
    //
    //
    //
    if (epsN < 0) {
        ////////////////////////////////////////////////////////////
        // Compressive contact;
        ////////////////////////////////////////////////////////////
        float sgmN = 0;
        float sgmM = 0;
        float sgmL = 0;
        float sgmT = 0;
        //

        float stot;
        if (epsN >= epsA) {
            // stot=epsQ*ENm;
            sgmN = epsN * ENm;
            contact_info_strain_y = 0;
            contact_info_strain_z = 0;
        } else {
            sgmN = (epsA)*ENm + (epsN - epsA) * ENa;
            sgmM = 0;
            sgmL = 0;
        }

        // std::cout<<"delta_new: "<<delta_new<<" epsN: "<<epsN<< " epsQ:
        // "<<epsQ<<" stot: "<<stot<<" sgmN: "<<sgmN<<std::endl;
        //////////////////////////////////////////////////////////
        //
        // Viscous stresses
        //
        //////////////////////////////////////////////////////////
        //
        // depsN=relvel_n_mag/Lij*dT;
        // depsT=relvel_t_mag/Lij*dT;
        // calculate strain rates
        float vdepsN = depsN / dT;
        float vdepsM = depsM / dT;
        float vdepsL = depsL / dT;
        float vDeps = powf((beta * vdepsN * vdepsN + vdepsM * vdepsM + vdepsL * vdepsL), 0.5);
        //
        // std::cout << "Lij : " << Lij << " vdeps_Comp : " << vDeps <<
        // std::endl;
        float eta;
        float lambda = lambda0;

        {
            // double lambda = rungeKutta4<double>(dfloc, lambda0, current_time,
            // flocbeta, vDeps, flocm, flocTcr, dT, 1); lambda=0;
            // map_contact_info[mykey].lambda=lambda;
            if (vDeps <= Deps0) {
                eta = eta0;
            } else {
                eta = eta_inf * powf(abs(vDeps), np - 1.) + sgmTau0 * (1.0 + lambda) / vDeps;
            }
        }

        float sgmN_vis = beta * eta * vdepsN;
        float sgmM_vis = eta * vdepsM;
        float sgmL_vis = eta * vdepsL;
        float sgmT_vis = powf(sgmM_vis * sgmM_vis + sgmL_vis * sgmL_vis, 0.5);
        //
        //
        //
        float forceN = contact_area * (sgmN + sgmN_vis);
        float forceM = contact_area * (sgmM + sgmM_vis);
        float forceL = contact_area * (sgmL + sgmL_vis);
        // std::cout<<"epsN "<<epsN<<" sgmN "<<sgmN<<std::endl;
        // force[0]=forceN;force[1]=contact_area*(sgmM-sgmM_vis);force[2]=contact_area*(sgmL-sgmL_vis);
        force = forceN * normal_dir + forceM * Vy + forceL * Vz;
        // std::cout<<" force "<<force<<std::endl;
        /*
        force = -forceN * normal_dir;
        if (relvel_t_mag >= sys.GetSlipVelocityThreshold())
                force -= (forceT / relvel_t_mag) * relvel_t;
    */
        // return -force;

    } else {
        ////////////////////////////////////////////////////////////
        // Tensile contact
        ////////////////////////////////////////////////////////////

        //////////////////////////////////////////////////////////
        //
        // Calculate Stress from material stiffness
        //
        //////////////////////////////////////////////////////////
        float sgmN = ENm * epsN;
        float sgmM = 0;
        float sgmL = 0;
        if (sgmN > sgmTmax)
            sgmN = sgmTmax;
        //
        //////////////////////////////////////////////////////////
        //
        // Viscous stresses
        //
        //////////////////////////////////////////////////////////
        //
        // calculate strain rates
        float vdepsN = depsN / dT;
        float vdepsM = depsM / dT;
        float vdepsL = depsL / dT;
        float vDeps = powf((beta * vdepsN * vdepsN + vdepsM * vdepsM + vdepsL * vdepsL), 0.5);
        //
        // std::cout << "Lij : " << Lij << " vdeps_Tension : " << vDeps <<
        // std::endl;
        float eta;
        float lambda = lambda0;

        {
            // double lambda = rungeKutta4<double>(dfloc, lambda0, current_time,
            // flocbeta, vDeps, flocm, flocTcr, dT, 1); lambda=0;
            // map_contact_info[mykey].lambda=lambda;
            if (vDeps <= Deps0) {
                eta = eta0;
            } else {
                eta = eta_inf * powf(abs(vDeps), np - 1.) + sgmTau0 * (1.0 + lambda) / vDeps;
            }
        }
        // std::cout<<"relvel"<<relvel<<" Deps0: "<<Deps0<<" vDeps"<<vDeps<<"
        // eta: "<<eta<<std::endl;
        float sgmN_vis = beta * eta * vdepsN;
        float sgmM_vis = eta * vdepsM;
        float sgmL_vis = eta * vdepsL;
        // double sgmT=powf(sgmM*sgmM+sgmL*sgmL,0.5);
        //////////////////////////////////////////////////////////
        //
        // Combine Viscous stresses and stiffness stresses and calculate forces
        //
        //////////////////////////////////////////////////////////
        // exit(0);
        float forceN = contact_area * (sgmN + sgmN_vis);
        float forceM = contact_area * (sgmM + sgmM_vis);
        float forceL = contact_area * (sgmL + sgmL_vis);
        // std::cout<<"epsN "<<epsN<<" sgmN "<<sgmN<<std::endl;
        // double forceT = sgmT * contact_area;
        // force[0]=forceN;force[1]=contact_area*(sgmM-sgmM_vis);force[2]=contact_area*(sgmL-sgmL_vis);
        force = forceN * normal_dir + forceM * Vy + forceL * Vz;
        /*
        force = -forceN * normal_dir;
        if (relvel_t_mag >= sys.GetSlipVelocityThreshold())
                force -= (forceT / relvel_t_mag) * relvel_t;
        */
        // return -force;
    }
    //   ChVector3d torque = Vcross((p1 - p0), -force);
    // return std::make_pair(-force, torque);
    //   return {-force, torque};

    // if (isnan(length(force))) {
    //     printf("Force NaN!!!!\n");
    // }
} else {
    // No contact
    contact_info_step_time = 0.f;
    contact_info_lambda = 0.f;
    contact_info_strain_x = 0.f;
    contact_info_strain_y = 0.f;
    contact_info_strain_z = 0.f;
    force = make_float3(0, 0, 0);
}
