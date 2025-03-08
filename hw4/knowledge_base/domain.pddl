;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Op-block-L world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Semantics
;;; position-1 = block L with top of "L" at 90 degrees
;;; position-2 = block L with top of "L" at 0 degrees
;;; position-3 = block L with top of "L" at 270  degrees
;;; position-4 = block L with top of "L" at 180  degrees
;;; rotate-1 = action to move block L at position-4 to position-1
;;; rotate-2 = action to move block L at position-1 to position-2
;;; rotate-3 = action to move block L at position-2 to position-3
;;; rotate-4 = action to move block L at position-3 to position-4
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain BLOCKS)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (position-0 ?x)
	       (position-1 ?x)
	       (position-2 ?x)
	       (position-3 ?x)
	       (position-4 ?x)
	       )

  (:action rotate-1
	     :parameters (?x - block)
	     :precondition (position-4 ?x)
	     :effect
	     (and (not (position-4 ?x))
		   (position-1 ?x))
		   )

    (:action rotate-2
	     :parameters (?x - block)
	     :precondition (position-1 ?x)
	     :effect
	     (and (not (position-1 ?x))
		   (position-2 ?x))
		   )

    (:action rotate-3
	     :parameters (?x - block)
	     :precondition (position-2 ?x)
	     :effect
	     (and (not (position-2 ?x))
		   (position-3 ?x))
		   )

    (:action rotate-4
	     :parameters (?x - block)
	     :precondition (position-3 ?x)
	     :effect
	     (and (not (position-3 ?x))
		   (position-4 ?x))
		   )

		   )
