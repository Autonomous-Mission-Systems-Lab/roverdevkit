import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

const PRIMARY_TARGETS: Array<{
  label: string;
  medianR2: string;
  coverage: string;
}> = [
  { label: "Range", medianR2: "0.980", coverage: "97.3%" },
  { label: "Energy margin", medianR2: "0.960", coverage: "93.1%" },
  { label: "Slope capability", medianR2: "0.988", coverage: "93.4%" },
  { label: "Total mass", medianR2: "1.000", coverage: "91.5%" },
];

export function AboutModelDialog({ children }: { children: React.ReactNode }) {
  return (
    <Dialog>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>About the RoverDevKit Tradespace Explorer</DialogTitle>
          <DialogDescription>
            RoverDevKit Tradespace Explorer predicts mission performance of a conceptual lunar micro-rover design with two cooperating
            models: a deterministic physics evaluator and a learned ML
            surrogate. The sections below describe what each model predicts,
            how it is built, and where it is used in the tool.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 text-sm leading-6">
          <section>
            <h3 className="font-medium text-[var(--color-foreground)]">
              Physics evaluator
            </h3>
            <p className="mt-1 text-[var(--color-muted-foreground)]">
              <span className="font-medium text-[var(--color-foreground)]">
                What it predicts.
              </span>{" "}
              For one rover design (12 variables) on one of the four canonical
              lunar mission scenarios, the evaluator runs a time-stepped
              traverse and returns four mission-level outputs: range,
              end-of-mission energy margin, statically achievable slope, and
              total mass. Range is reported as an <em>energy-feasible</em>{" "}
              distance — when the battery hits its depth-of-discharge floor the
              simulator throttles drive duty to whatever fraction the
              instantaneous solar input can sustain, so under-powered designs
              no longer report their full kinematic envelope.
            </p>
            <p className="mt-2 text-[var(--color-muted-foreground)]">
              <span className="font-medium text-[var(--color-foreground)]">
                How it is built.
              </span>{" "}
              The traverse loop composes open sub-models from
              published sources: Bekker&ndash;Wong terramechanics with an
              engaged-grouser shear-thrust term (Iizuka &amp; Kubota 2011), a
              Larson&ndash;Wertz solar / battery state-of-charge model, a
              parametric mass-estimating relationship fit to flown lunar
              micro-rovers (SMAD MER methodology), and a lumped-parameter
              thermal-survival check (Gilmore 2002). 
            </p>
            <p className="mt-2 text-[var(--color-muted-foreground)]">
              <span className="font-medium text-[var(--color-foreground)]">
                Wheel&ndash;soil model.
              </span>{" "}
              Inside each mission, wheel forces are computed analytically with
              Bekker&ndash;Wong, augmented with an engaged-grouser shear-thrust
              term that scales tractive shear stress with grouser height and
              count up to a saturation cap matching the Iizuka &amp; Kubota
              lab-reported asymptote.
            </p>
            <p className="mt-3 text-[var(--color-muted-foreground)]">
              <span className="font-medium text-[var(--color-foreground)]">
                Where it is used.
              </span>{" "}
              The evaluator is the ground-truth oracle for every quantitative
              answer the tool gives. It produces the deterministic median
              values on the Current Design tab, the cell evaluations on the
              Parametric Sweep tab, and the NSGA-II fitness function on the
              Optimize Design tab. It is also the source of the training data
              for the surrogate.
            </p>
          </section>

          <section>
            <h3 className="font-medium text-[var(--color-foreground)]">
              ML surrogate
            </h3>
            <p className="mt-1 text-[var(--color-muted-foreground)]">
              <span className="font-medium text-[var(--color-foreground)]">
                What it predicts.
              </span>{" "}
              For the same design &times; scenario inputs as the evaluator,
              the surrogate predicts a calibrated 90% prediction interval
              around each of the four primary metrics. The deterministic
              median shown in the UI comes from the evaluator; the surrogate
              supplies the q05&ndash;q95 envelope rendered as a blue bar
              around that median.
            </p>
            <p className="mt-2 text-[var(--color-muted-foreground)]">
              <span className="font-medium text-[var(--color-foreground)]">
                How it is built.
              </span>{" "}
              40,000 candidate rovers were generated by Latin-Hypercube
              sampling over the 12-D design space and evaluated on the four
              canonical scenarios with the deterministic physics evaluator.
              The resulting corpus (~32k feasible rows after motor-torque
              screening) is split 80/10/10 for training, validation, and test.
              The surrogate fits three quantile XGBoost regressors per metric
              at &tau; = 0.05, 0.50, and 0.95. Hyperparameters were tuned with
              Optuna on a squared-error baseline and reused across the three
              quantile heads with the loss switched to pinball. Results below
              are on the held-out test split, after row-wise monotone repair
              of the (q05, q50, q95) triple.
            </p>
            <div className="mt-3 overflow-hidden rounded-md border">
              <table className="w-full text-xs">
                <thead className="bg-[var(--color-muted)] text-[var(--color-muted-foreground)]">
                  <tr>
                    <th className="px-3 py-1.5 text-left font-medium">
                      Target
                    </th>
                    <th className="px-3 py-1.5 text-right font-medium">
                      Median R²
                    </th>
                    <th className="px-3 py-1.5 text-right font-medium">
                      90% PI coverage
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {PRIMARY_TARGETS.map((row) => (
                    <tr key={row.label} className="border-t">
                      <td className="px-3 py-1.5">{row.label}</td>
                      <td className="px-3 py-1.5 text-right tabular-nums">
                        {row.medianR2}
                      </td>
                      <td className="px-3 py-1.5 text-right tabular-nums">
                        {row.coverage}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-[var(--color-muted-foreground)]">
              Coverage is empirical. Energy margin, slope capability, and total
              mass land within a few percentage points of the 90% nominal,
              with mild over-coverage indicating modestly conservative bands
              rather than tight ones. Range over-covers more noticeably
              (~97%) because the achievable-distance surface kinks at the
              energy floor where the duty-cycle throttle engages, so the
              quantile heads compensate by widening intervals near that
              boundary.
            </p>
            <p className="mt-3 text-[var(--color-muted-foreground)]">
              <span className="font-medium text-[var(--color-foreground)]">
                Where it is used.
              </span>{" "}
              The surrogate supplies the 90% prediction intervals around the
              evaluator&rsquo;s deterministic median on the Current Design
              tab and powers the TreeSHAP feature attributions on the Explain
              Design tab. NSGA-II on the Optimize Design tab calls the
              evaluator directly rather than the surrogate, because the
              corrected evaluator is fast enough to drive a live Pareto
              search and using evaluator-truth as the fitness function avoids
              any surrogate-approximation error on the optimization frontier.
            </p>
          </section>

          <section>
            <h3 className="font-medium text-[var(--color-foreground)]">
              Limitations
            </h3>
            <ul className="mt-1 list-disc space-y-1 pl-5 text-[var(--color-muted-foreground)]">
              <li>
                Predictions are valid inside the calibrated design space and
                the four canonical scenarios; out-of-range design inputs are
                rejected by the form.
              </li>
              <li>
                Real-flown rovers (e.g. Pragyan, Yutu-2, MoonRanger,
                Rashid-1) are recovered well on mass and slope capability,
                but range and energy margin are noisier because those rovers
                run mission profiles outside the canonical scenario set.
              </li>
              <li>
                The Bekker&ndash;Wong wheel-soil model carries the published
                ~15&ndash;30% model-form error for single-wheel drawbar pull
                and sinkage, so absolute mobility accuracy is bounded
                accordingly.
              </li>
            </ul>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}
