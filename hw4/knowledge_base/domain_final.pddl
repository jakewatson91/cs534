;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Op-block-L world
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Semantics
;;; position-1 = block L with top of "L" at 90 degrees
;;; position-2 = block L with top of "L" at 0 degrees
;;; position-3 = block L with top of "L" at 270  degrees
;;; position-4 = block L with top of "L" at 180  degrees
;;; position-5 = block with count-2 blocks rotated 90 degress clockwise
;;; rotate-1 = action to move block L at position-4 to position-1
;;; rotate-2 = action to move block L at position-1 to position-2
;;; rotate-3 = action to move block L at position-2 to position-3
;;; rotate-4 = action to move block L at position-3 to position-4
;;; rotate-5 = action to move block at position-5 to count-3

;;; count-1 = there is one block (one cell) at (1, 1)
;;; count-2 = there are two blocks (two cells) at (1, 1) and (1, 3)
;;; count-3 = there are two blocks (two cells) as (1, 3) and (3, 3)
;;; create-1 = add one block at (1, 3) if there is only one block
;;; delete-1 = delete one block if there are two blocks at (1, 1) and (1, 3)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (domain BLOCKS)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (position-0 ?x)
	       (position-1 ?x)
	       (position-2 ?x)
	       (position-3 ?x)
	       (position-4 ?x)
           (position-5 ?x)
           (count-1 ?x)
	       (count-2 ?x)
           (count-3 ?x)
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
    
    (:action rotate-5
	     :parameters (?x - block)
	     :precondition (count-2 ?x)
	     :effect (and 
            (not (count-2 ?x))
            (position-5 ?x))
		)

    (:action create-1
     :parameters (?x - block)
     :precondition (count-1 ?x)  ;; only create if there's one block
     :effect (and 
         (not (count-1 ?x)) ;; remove one block state
         (count-2 ?x))  ;; add second block
        )

    ;; Delete the second block, going back to only one block
    (:action delete-1
        :parameters (?x - block)
        :precondition (count-2 ?x)  ;; only delete if there are two blocks
        :effect (and 
            (not (count-2 ?x))  ;; remove two blocks state
            (count-1 ?x))  ;; back to one block
        )
    )
